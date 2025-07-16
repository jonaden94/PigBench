import cv2
import torchvision.transforms.functional as F
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, video_path: str, short_max_size: int, long_max_size: int):
        video_path = video_path
        self.video_cap = cv2.VideoCapture(video_path)
        self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.short_max_size = short_max_size
        self.long_max_size = long_max_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        return

    def load(self, ):
        ret, image = self.video_cap.read()
        assert image is not None
        return image

    def process_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_image = image.copy()
        h, w = image.shape[:2]
        scale = self.short_max_size / min(h, w)
        if max(h, w) * scale > self.long_max_size:
            scale = self.long_max_size / max(h, w)
        target_h = int(h * scale)
        target_w = int(w * scale)
        image = cv2.resize(image, (target_w, target_h))
        image = F.normalize(F.to_tensor(image), self.mean, self.std)
        return image, ori_image

    def __getitem__(self, item):
        image = self.load()
        return self.process_image(image=image)

    def __len__(self):
        return self.frame_count
