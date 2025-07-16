import os
import cv2
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, video_path: str, bgr_to_rgb: bool = False):
        video_path = video_path
        self.video_cap = cv2.VideoCapture(video_path)
        self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.bgr_to_rgb = bgr_to_rgb

    def load(self):
        ret, image = self.video_cap.read()
        assert image is not None
        return image

    def __getitem__(self, item):
        image = self.load()
        image_ori = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.bgr_to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_ori = image.copy()
        return image, image_ori

    def __len__(self):
        return self.frame_count


class SeqDataset(Dataset):
    def __init__(self, seq_dir: str, bgr_to_rgb: bool = False):
        image_paths = sorted(os.listdir(os.path.join(seq_dir, "img1")))
        image_paths = [os.path.join(seq_dir, "img1", _) for _ in image_paths if ("jpg" in _) or ("png" in _)]
        self.image_paths = image_paths
        self.bgr_to_rgb = bgr_to_rgb

    @staticmethod
    def load(path):
        image = cv2.imread(path)
        assert image is not None
        return image

    def __getitem__(self, item):
        image = self.load(self.image_paths[item])
        image_ori = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.bgr_to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_ori = image.copy()
        return image, image_ori

    def __len__(self):
        return len(self.image_paths)
    