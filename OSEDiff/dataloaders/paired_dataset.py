import glob
import os

from PIL import Image
from torch.utils import data
from torchvision import transforms


class PairedCaptionDataset(data.Dataset):
    """Loads GT images and their positive/negative exposure prompts.

    Expected directory layout for each root folder:

        root/
          gt/
            image_001.png
          tag/
            image_001_+2.txt
            image_001_-2.txt
    """

    def __init__(self, root_folders, args=None):
        super().__init__()
        self.args = args
        self.gt_list = []
        self.prompt_path_map_list = []

        if isinstance(root_folders, str):
            root_folders = [item.strip() for item in root_folders.split(",") if item.strip()]

        for root_folder in root_folders:
            gt_dir = os.path.join(root_folder, "gt")
            tag_dir = os.path.join(root_folder, "tag")

            if not os.path.isdir(gt_dir):
                print(f"Warning: GT directory not found: {gt_dir}")
                continue

            for gt_image_path in sorted(glob.glob(os.path.join(gt_dir, "*.png"))):
                base_name = os.path.splitext(os.path.basename(gt_image_path))[0]
                pos_prompt_path = os.path.join(tag_dir, f"{base_name}_+2.txt")
                neg_prompt_path = os.path.join(tag_dir, f"{base_name}_-2.txt")

                if os.path.exists(pos_prompt_path) and os.path.exists(neg_prompt_path):
                    self.gt_list.append(gt_image_path)
                    self.prompt_path_map_list.append(
                        {
                            "pos": pos_prompt_path,
                            "neg": neg_prompt_path,
                        }
                    )

        if not self.gt_list:
            raise ValueError(
                "No valid training samples found. Each root must contain gt/*.png images "
                "and matching tag/<image_stem>_+2.txt and tag/<image_stem>_-2.txt files."
            )

        print(f"Loaded {len(self.gt_list)} training images.")
        self.img_preproc = transforms.Compose([transforms.ToTensor()])

    @staticmethod
    def _read_prompt_file(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()

    def __getitem__(self, index):
        gt_path = self.gt_list[index]
        prompt_paths = self.prompt_path_map_list[index]

        gt_img = Image.open(gt_path).convert("RGB")
        gt_tensor_0_1 = self.img_preproc(gt_img)

        return {
            "output_pixel_values": gt_tensor_0_1,
            "prompt+": self._read_prompt_file(prompt_paths["pos"]),
            "prompt-": self._read_prompt_file(prompt_paths["neg"]),
            "neg_prompt": self.args.neg_prompt,
        }

    def __len__(self):
        return len(self.gt_list)
