import torch
from diffusers import DDPMScheduler
from peft import LoraConfig
from transformers import AutoTokenizer, CLIPTextModel

from models.autoencoder_kl import AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel


def initialize_vae(args):
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.train()

    target_modules = []
    target_patterns = [
        "conv1",
        "conv2",
        "conv_in",
        "conv_shortcut",
        "conv",
        "conv_out",
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
    ]
    for name, _param in vae.named_parameters():
        if "bias" in name or "norm" in name:
            continue
        for pattern in target_patterns:
            if pattern in name and "encoder" in name:
                target_modules.append(name.replace(".weight", ""))
            elif "quant_conv" in name and "post_quant_conv" not in name:
                target_modules.append(name.replace(".weight", ""))

    lora_config = LoraConfig(
        r=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    vae.add_adapter(lora_config, adapter_name="default_encoder")
    return vae, target_modules


def initialize_unet(args):
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    encoder_modules = []
    decoder_modules = []
    other_modules = []
    target_patterns = [
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
        "conv",
        "conv1",
        "conv2",
        "conv_in",
        "conv_shortcut",
        "conv_out",
        "proj_out",
        "proj_in",
        "ff.net.2",
        "ff.net.0.proj",
    ]
    for name, _param in unet.named_parameters():
        if "bias" in name or "norm" in name:
            continue
        for pattern in target_patterns:
            if pattern in name and ("down_blocks" in name or "conv_in" in name):
                encoder_modules.append(name.replace(".weight", ""))
                break
            if pattern in name and ("up_blocks" in name or "conv_out" in name):
                decoder_modules.append(name.replace(".weight", ""))
                break
            if pattern in name:
                other_modules.append(name.replace(".weight", ""))
                break

    unet.add_adapter(
        LoraConfig(r=args.lora_rank, init_lora_weights="gaussian", target_modules=encoder_modules),
        adapter_name="default_encoder",
    )
    unet.add_adapter(
        LoraConfig(r=args.lora_rank, init_lora_weights="gaussian", target_modules=decoder_modules),
        adapter_name="default_decoder",
    )
    unet.add_adapter(
        LoraConfig(r=args.lora_rank, init_lora_weights="gaussian", target_modules=other_modules),
        adapter_name="default_others",
    )

    return unet, encoder_modules, decoder_modules, other_modules


class OSEDiff_gen(torch.nn.Module):
    def __init__(self, args, device=None):
        super().__init__()
        self.args = args
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
        ).to(self.device)
        self.text_encoder.requires_grad_(False)

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="scheduler",
        )
        self.noise_scheduler.set_timesteps(1, device=self.device)
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)

        self.vae, self.lora_vae_modules_encoder = initialize_vae(args)
        (
            self.unet,
            self.lora_unet_modules_encoder,
            self.lora_unet_modules_decoder,
            self.lora_unet_others,
        ) = initialize_unet(args)

        self.vae.to(self.device)
        self.unet.to(self.device)
        self.timesteps = torch.tensor([999], device=self.device).long()
        self.lora_rank_unet = args.lora_rank
        self.lora_rank_vae = args.lora_rank

    def set_train(self):
        self.unet.train()
        self.vae.train()
        for name, param in self.unet.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for name, param in self.vae.named_parameters():
            if "lora" in name:
                param.requires_grad = True

    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption,
                    max_length=self.tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids
                prompt_embeds = self.text_encoder(text_input_ids.to(self.text_encoder.device))[0]
                prompt_embeds_list.append(prompt_embeds)
        return torch.concat(prompt_embeds_list, dim=0)

    def forward(self, c_t, batch=None):
        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        prompt_embeds = self.encode_prompt(batch["prompt"])
        neg_prompt_embeds = self.encode_prompt(batch["neg_prompt"])

        model_pred = self.unet(
            encoded_control,
            self.timesteps,
            encoder_hidden_states=prompt_embeds.to(torch.float32),
        ).sample
        x_denoised = self.noise_scheduler.step(
            model_pred,
            self.timesteps,
            encoded_control,
            return_dict=True,
        ).prev_sample
        output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample.clamp(-1, 1)
        return output_image, x_denoised, prompt_embeds, neg_prompt_embeds

    def save_model(self, output_path):
        state = {
            "vae_lora_encoder_modules": self.lora_vae_modules_encoder,
            "unet_lora_encoder_modules": self.lora_unet_modules_encoder,
            "unet_lora_decoder_modules": self.lora_unet_modules_decoder,
            "unet_lora_others_modules": self.lora_unet_others,
            "rank_unet": self.lora_rank_unet,
            "rank_vae": self.lora_rank_vae,
            "state_dict_unet": {
                key: value
                for key, value in self.unet.state_dict().items()
                if "lora" in key or "conv_in" in key
            },
            "state_dict_vae": {
                key: value for key, value in self.vae.state_dict().items() if "lora" in key
            },
        }
        torch.save(state, output_path)
