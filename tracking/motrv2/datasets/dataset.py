# # ------------------------------------------------------------------------
# # Modified from MOTRv2 (https://github.com/megvii-research/MOTRv2)
# # Copyright (c) 2022 megvii-research. All Rights Reserved.
# # ------------------------------------------------------------------------
# # Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# # Copyright (c) 2020 SenseTime. All Rights Reserved.
# # ------------------------------------------------------------------------
# # Modified from DETR (https://github.com/facebookresearch/detr)
# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# # ------------------------------------------------------------------------

"""
MOT dataset which returns image_id for evaluation.
"""
from collections import defaultdict
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import copy
import datasets.transforms as T
from models.structures import Instances
from random import randint


class DetMOTDetection:
    def __init__(self, cfg, transform):
        self.transform = transform
        self.num_frames_per_batch = max(cfg.sampler_lengths)
        self.aug_random_shift_max_ratio = cfg.aug_random_shift_max_ratio
        self.sample_mode = cfg.sample_mode
        self.sample_interval = cfg.sample_interval
        self.video_dict = {}
        
        # data folders
        self.data_root = cfg.data_root
        self.mot_datasets = cfg.mot_datasets
        self.mot_dataset_splits = cfg.mot_dataset_splits
        self.det_datasets = cfg.det_datasets
        self.det_dataset_splits = cfg.det_dataset_splits

        # add labels of MOT datasets
        self.labels_full = defaultdict(lambda : defaultdict(list))
        for dataset, dataset_split in zip(self.mot_datasets, self.mot_dataset_splits):
            split_dir = os.path.join(dataset, dataset_split)
            self.add_mot_folder(split_dir)
        vid_files = list(self.labels_full.keys())

        # extract info for MOT videos
        self.indices = []
        self.vid_tmax = {}
        for vid in vid_files:
            self.video_dict[vid] = len(self.video_dict)
            t_min = min(self.labels_full[vid].keys())
            t_max = max(self.labels_full[vid].keys()) + 1
            self.vid_tmax[vid] = t_max - 1
            for t in range(t_min, t_max - self.num_frames_per_batch):
                self.indices.append((vid, t))
        print(f"Using {len(vid_files)} videos ({len(self.indices)} frames) from video datasets", flush=True)

        self.sampler_steps: list = cfg.sampler_steps
        self.lengths: list = cfg.sampler_lengths
        print("sampler_steps={} lenghts={}".format(self.sampler_steps, self.lengths), flush=True)

        # extra detection data
        self.det_indices = []
        if cfg.append_det:
            for det_dataset, det_dataset_split in zip(self.det_datasets, self.det_dataset_splits):
                det_dir = os.path.join(self.data_root, det_dataset, det_dataset_split)
                gt_dir = os.path.join(det_dir, 'gts')
                for gt_file_name in os.listdir(gt_dir):
                    ID = gt_file_name[:-4]
                    boxes = []
                    gt_file_path = os.path.join(gt_dir, gt_file_name)
                    with open(gt_file_path, "r") as gt_file:
                        for line in gt_file:
                            line = line[:-1]
                            _, _, x, y, w, h = line.split(" ")
                            boxes.append([float(x), float(y), float(w), float(h)])
                    self.det_indices.append((ID, det_dataset, det_dataset_split, boxes))
                
        print(f"Using {len(self.det_indices)} images from detection datasets", flush=True)

        if cfg.det_db:
            with open(cfg.det_db) as f:
                self.det_db = json.load(f)
        else:
            self.det_db = defaultdict(list)

    def add_mot_folder(self, split_dir):
        print("Adding", split_dir, flush=True)
        for vid in os.listdir(os.path.join(self.data_root, split_dir)):
            vid = os.path.join(split_dir, vid)
            gt_path = os.path.join(self.data_root, vid, 'gt', 'gt.txt')
            for l in open(gt_path):
                t, i, *xywh, mark, label = l.strip().split(',')[:8]
                t, i, mark, label = map(int, (t, i, mark, label))
                if mark == 0: # always 1 in dancetrack and pigtrack
                    continue
                else:
                    crowd = False
                x, y, w, h = map(float, (xywh))
                self.labels_full[vid][t].append([x, y, w, h, i, crowd])
                    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            # fixed sampling length.
            return

        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                lengths_idx = i
        print("set epoch: epoch {} sampler_lengths={}".format(epoch, self.lengths[lengths_idx]), flush=True)
        self.num_frames_per_batch = self.lengths[lengths_idx]

    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch), flush=True)
        self.set_epoch(self.current_epoch + 1)

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        n_gt = len(targets['labels'])
        gt_instances.boxes = targets['boxes'][:n_gt]
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        return gt_instances

    def load_det(self, index):
        ID, dataset, split, boxes = self.det_indices[index]
        det_dir = os.path.join(self.data_root, dataset, split)
        boxes = copy.deepcopy(boxes)
        img = Image.open(os.path.join(det_dir, 'images', f'{ID}.jpg'))

        w, h = img._size
        n_gts = len(boxes)
        scores = [0. for _ in range(len(boxes))]
        for line in self.det_db[f'DetDataset/{ID}']:
            *box, s = map(float, line.split(','))
            boxes.append(box)
            scores.append(s)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        areas = boxes[..., 2:].prod(-1)
        boxes[:, 2:] += boxes[:, :2]

        target = {
            'boxes': boxes,
            'scores': torch.as_tensor(scores),
            'labels': torch.zeros((n_gts, ), dtype=torch.long),
            'iscrowd': torch.zeros((n_gts, ), dtype=torch.bool),
            'image_id': torch.tensor([0]),
            'area': areas,
            'obj_ids': torch.arange(n_gts),
            'size': torch.as_tensor([h, w]),
            'orig_size': torch.as_tensor([h, w]),
            'dataset': "DetectionDataset",
        }
        rs = T.FixedMotRandomShift(self.num_frames_per_batch, self.aug_random_shift_max_ratio)
        return rs([img], [target])

    def _pre_single_frame(self, vid, idx: int):
        img_path = os.path.join(self.data_root, vid, 'img1', f'{idx:08d}.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(self.data_root, vid, 'img1', f'{idx:04d}.jpg')
        img = Image.open(img_path)
        targets = {}
        w, h = img._size
        assert w > 0 and h > 0, "invalid image {} with shape {} {}".format(img_path, w, h)
        obj_idx_offset = self.video_dict[vid] * 100000  # 100000 unique ids is enough for a video.

        targets['dataset'] = 'MOT'
        targets['boxes'] = []
        targets['iscrowd'] = []
        targets['labels'] = []
        targets['obj_ids'] = []
        targets['scores'] = []
        targets['image_id'] = torch.as_tensor(idx)
        targets['size'] = torch.as_tensor([h, w])
        targets['orig_size'] = torch.as_tensor([h, w])
        for *xywh, id, crowd in self.labels_full[vid][idx]:
            targets['boxes'].append(xywh)
            assert not crowd
            targets['iscrowd'].append(crowd)
            targets['labels'].append(0)
            targets['obj_ids'].append(id + obj_idx_offset)
            targets['scores'].append(1.)
        det_db_key = os.path.join(vid.split('/')[2], f'{idx:08d}')
        for line in self.det_db[det_db_key]:
            *box, s = map(float, line.split(','))
            targets['boxes'].append(box)
            targets['scores'].append(s)

        targets['iscrowd'] = torch.as_tensor(targets['iscrowd'])
        targets['labels'] = torch.as_tensor(targets['labels'])
        targets['obj_ids'] = torch.as_tensor(targets['obj_ids'], dtype=torch.float64)
        targets['scores'] = torch.as_tensor(targets['scores'])
        targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)
        targets['boxes'][:, 2:] += targets['boxes'][:, :2]
        return img, targets

    def _get_sample_range(self, start_idx):
        # take default sampling method for normal dataset.
        assert self.sample_mode in ['fixed_interval', 'random_interval'], 'invalid sample mode: {}'.format(self.sample_mode)
        if self.sample_mode == 'fixed_interval':
            sample_interval = self.sample_interval
        elif self.sample_mode == 'random_interval':
            sample_interval = np.random.randint(1, self.sample_interval + 1)
        default_range = start_idx, start_idx + (self.num_frames_per_batch - 1) * sample_interval + 1, sample_interval
        return default_range

    def pre_continuous_frames(self, vid, indices):
        return zip(*[self._pre_single_frame(vid, i) for i in indices])

    def sample_indices(self, vid, f_index):
        assert self.sample_mode == 'random_interval'
        rate = randint(1, self.sample_interval + 1)
        tmax = self.vid_tmax[vid]
        ids = [f_index + rate * i for i in range(self.num_frames_per_batch)]
        return [min(i, tmax) for i in ids]

    def __getitem__(self, idx):
        if idx < len(self.indices):
            vid, f_index = self.indices[idx]
            indices = self.sample_indices(vid, f_index)
            images, targets = self.pre_continuous_frames(vid, indices)
        else:
            images, targets = self.load_det(idx - len(self.indices))
        if self.transform is not None:
            images, targets = self.transform(images, targets)
        gt_instances, proposals = [], []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])
            gt_instances.append(gt_instances_i)
            n_gt = len(targets_i['labels'])
            proposals.append(torch.cat([
                targets_i['boxes'][n_gt:],
                targets_i['scores'][n_gt:, None],
            ], dim=1))
        return {
            'imgs': images,
            'gt_instances': gt_instances,
            'proposals': proposals,
        }

    def __len__(self):
        return len(self.indices) + len(self.det_indices)

# determine if annotation is crowd (basically invalid)
def is_crowd(ann):
    return 'extra' in ann and 'ignore' in ann['extra'] and ann['extra']['ignore'] == 1

def make_transforms(cfg):
    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transforms = T.MotCompose([
        T.MotRandomHorizontalFlip(),
        T.MotRandomSelect(
            T.MotRandomResize(cfg.aug_resize_scales, max_size=cfg.aug_max_size),
            T.MotCompose([
                T.MotRandomResize(cfg.aug_random_resize),
                T.FixedMotRandomCrop(cfg.aug_random_crop_min, cfg.aug_random_crop_max),
                T.MotRandomResize(cfg.aug_resize_scales, max_size=cfg.aug_max_size),
            ])
        ),
        T.MOTHSV(),
        T.MotReverseClip(cfg.aug_reverse_clip),
        normalize,
    ])
    return transforms

def build_dataset(cfg):
    root = Path(cfg.data_root)
    assert Path(cfg.data_root).exists(), f'provided root path for data "{root}" does not exist'
    transform = make_transforms(cfg)
    dataset = DetMOTDetection(cfg, transform=transform)
    print("Dataset succesfully loaded!", flush=True)
    return dataset
