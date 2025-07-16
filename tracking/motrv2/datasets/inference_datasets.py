import os
import torchvision.transforms.functional as F
import torch
import cv2
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, video_path, det_db, short_max_size, long_max_size):
        super().__init__()
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)[:-4]
        self.det_db = det_db

        self.short_max_size = short_max_size
        self.long_max_size = long_max_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.video_cap = cv2.VideoCapture(self.video_path)
        self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def load(self):
        ret, image = self.video_cap.read()
        assert image is not None, "Failed to load frame from video."
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
        image = image.unsqueeze(0)
        return image, ori_image

    def __len__(self):
        return self.frame_count

    def __getitem__(self, item):
        image = self.load()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # load bbox prior
        det_db_key = f"{self.video_name}/{item:08d}"
        proposals = []
        for line in self.det_db[det_db_key]:
            l, t, w, h, s = list(map(float, line.split(',')))
            proposals.append([(l + w / 2) / image.shape[1],
                                (t + h / 2) / image.shape[0],
                                w / image.shape[1],
                                h / image.shape[0],
                                s])

        proposals = torch.as_tensor(proposals).reshape(-1, 5)
        return self.process_image(image=image) + (proposals,)


class ListImgDataset(Dataset):
    def __init__(self, data_root, img_path_list, det_db, short_max_size, long_max_size) -> None:
        super().__init__()
        self.data_root = data_root
        self.img_path_list = img_path_list
        self.det_db = det_db

        '''
        common settings
        '''
        self.short_max_size = short_max_size
        self.long_max_size = long_max_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_img_from_file(self, f_path):
        cur_img = cv2.imread(f_path)
        assert cur_img is not None, f_path
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        proposals = []
        im_h, im_w = cur_img.shape[:2]
        det_db_key = (f_path[len(self.data_root):-4]).lstrip(r"/\\")
        det_db_key = os.path.join(det_db_key.split('/')[2], det_db_key.split('/')[4])
        for line in self.det_db[det_db_key]:
            l, t, w, h, s = list(map(float, line.split(',')))
            proposals.append([(l + w / 2) / im_w,
                                (t + h / 2) / im_h,
                                w / im_w,
                                h / im_h,
                                s])
        return cur_img, torch.as_tensor(proposals).reshape(-1, 5)

    def init_img(self, img):
        ori_img = img.copy()
        seq_h, seq_w = img.shape[:2]
        scale = self.short_max_size / min(seq_h, seq_w)
        if max(seq_h, seq_w) * scale > self.long_max_size:
            scale = self.long_max_size / max(seq_h, seq_w)
        target_h = int(seq_h * scale)
        target_w = int(seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img

    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, index):
        img, proposals = self.load_img_from_file(self.img_path_list[index])
        return self.init_img(img) + (proposals,)
