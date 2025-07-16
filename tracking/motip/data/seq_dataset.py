# Modified from MOTIP (https://github.com/MCG-NJU/MOTIP)
# Copyright (c) Ruopeng Gao. All Rights Reserved.

import os
import cv2
import torchvision.transforms.functional as F
from torch.utils.data import Dataset


class SeqDataset(Dataset):
    def __init__(self, seq_dir: str, short_max_size: int, long_max_size: int):
        """
        Args:
            seq_dir:
            dataset: DanceTrack format or MOT17 format
        """
        image_paths = sorted(os.listdir(os.path.join(seq_dir, "img1")))
        image_paths = [os.path.join(seq_dir, "img1", _) for _ in image_paths if ("jpg" in _) or ("png" in _)]
        self.image_paths = image_paths
        self.short_max_size = short_max_size
        self.long_max_size = long_max_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        return

    @staticmethod
    def load(path):
        """
        Args:
            path:

        Returns:
        """
        image = cv2.imread(path)
        assert image is not None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def process_image(self, image):
        ori_image = image.copy()
        h, w = image.shape[:2]
        scale = self.short_max_size / min(h, w)
        if max(h, w) * scale > self.long_max_size:
            scale = self.long_max_size / max(h, w)
        target_h = int(h * scale)
        target_w = int(w * scale)
        image = cv2.resize(image, (target_w, target_h))
        image = F.normalize(F.to_tensor(image), self.mean, self.std)
        # image = image.unsqueeze(0)
        return image, ori_image

    def __getitem__(self, item):
        image = self.load(self.image_paths[item])
        return self.process_image(image=image)

    def __len__(self):
        return len(self.image_paths)
