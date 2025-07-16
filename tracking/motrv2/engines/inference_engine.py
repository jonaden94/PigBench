# ------------------------------------------------------------------------
# Modified from MOTRv2 (https://github.com/megvii-research/MOTRv2)
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from copy import deepcopy
import json
import os
import torchvision.transforms.functional as F
import torch
import time
import cv2
import numpy as np
from models import build_model
from util.tool import load_model
from util.misc import get_rank, get_world_size
from models.structures import Instances
from datasets.inference_datasets import ListImgDataset, VideoDataset
from mmdet.utils import register_all_modules
from mmdet.apis import init_detector, inference_detector
from mmengine.logging import MMLogger
import logging
MMLogger.get_instance("mmengine").setLevel(logging.WARNING)
logging.getLogger("mmengine").setLevel(logging.WARNING)


def get_bbox_priors(cfg, video_path, save_path_det_db):
    # Initialize model
    register_all_modules(init_default_scope=False)
    device = cfg.gpu if cfg.distributed else ('cuda' if torch.cuda.is_available() else 'cpu')
    model = init_detector(cfg.inference_detector_config, cfg.inference_detector_checkpoint, device=device)
    det_db = {}

    # Open video file
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    frame_idx = 0
    while True:
        ret, frame = video_cap.read()
        if not ret:
            break  # End of video

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference
        result = inference_detector(model, frame_rgb)
        scores = result.pred_instances.scores.cpu().numpy()
        bboxes = result.pred_instances.bboxes.cpu().numpy()

        # Filter boxes by score
        bboxes = bboxes[scores >= cfg.inference_detector_min_conf]
        scores = scores[scores >= cfg.inference_detector_min_conf]

        # Prepare detections
        detections = []
        for bbox, score in zip(bboxes, scores):
            res = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1], score]
            res = [str(x) for x in res]
            res = ",".join(res) + "\n"
            detections.append(res)

        # Create key for current frame
        video_name = os.path.basename(video_path)[:-4]
        image_name = f"{frame_idx:08d}"
        key = f"{video_name}/{image_name}"
        det_db[key] = detections
        
        if cfg.inference_save_video_frames:
            save_dir_frames = os.path.join(cfg.outputs_dir_results, "frames", video_name)
            os.makedirs(save_dir_frames, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir_frames, f"{frame_idx:08d}.jpg"), frame_rgb)
        frame_idx += 1
        
    # Release video capture
    video_cap.release()

    # Save updated detection database
    os.makedirs(os.path.dirname(save_path_det_db), exist_ok=True)
    with open(save_path_det_db, "w") as outfile:
        json.dump(det_db, outfile, indent=4)
        
    
class Detector(object):
    def __init__(self, cfg, model, vid_path, outputs_dir_results):
        self.cfg = cfg
        self.detr = model
        self.vid_path = vid_path
        self.seq_name = os.path.basename(vid_path)
            
        # get datasets
        if cfg.mode == 'inference' or cfg.mode == 'train':
            with open(self.cfg.det_db) as f:
                det_db = json.load(f)
            img_path_list = os.listdir(os.path.join(vid_path, 'img1'))
            img_path_list = [os.path.join(vid_path, 'img1', i) for i in img_path_list if 'jpg' in i]
            img_path_list = sorted(img_path_list)
            self.dataset = ListImgDataset(self.cfg.data_root, img_path_list, det_db, self.cfg.inference_short_max_size, self.cfg.inference_long_max_size)
        elif cfg.mode == 'video_inference':
            self.seq_name = self.seq_name[:-4]
            with open(self.cfg.det_db) as f:
                det_db = json.load(f)
            self.dataset = VideoDataset(self.vid_path, det_db, self.cfg.inference_short_max_size, self.cfg.inference_long_max_size)
            
        # make save directories and paths
        self.outputs_dir_inference = os.path.join(outputs_dir_results, 'tracker')
        self.outputs_dir_visualization = os.path.join(outputs_dir_results, 'visualization')
        os.makedirs(self.outputs_dir_inference, exist_ok=True)
        os.makedirs(self.outputs_dir_visualization, exist_ok=True)
        self.save_video_path = os.path.join(self.outputs_dir_visualization, self.seq_name + '.mp4')
        self.video_writer = None
        self.colors = (np.random.rand(64, 3) * 255).astype(dtype=np.int32)

    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float) -> Instances:
        keep = dt_instances.scores > prob_threshold
        keep &= dt_instances.obj_idxes >= 0
        return dt_instances[keep]

    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return dt_instances[keep]

    def detect(self, prob_threshold, area_threshold, draw_res=False):
        track_instances = None
        lines = []
        for i in range(self.dataset.__len__()):
            cur_img, ori_img, proposals = self.dataset.__getitem__(i)
            cur_img, proposals = cur_img.cuda(), proposals.cuda()

            if track_instances is not None:
                track_instances.remove('boxes')
                track_instances.remove('labels')
            seq_h, seq_w, _ = ori_img.shape

            res = self.detr.inference_single_image(cur_img, (seq_h, seq_w), track_instances, proposals)
            track_instances = res['track_instances']

            # filter det instances by score.
            dt_instances = deepcopy(track_instances)
            dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold)
            dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)

            bbox_xyxy = dt_instances.boxes.tolist()
            identities = dt_instances.obj_idxes.tolist()

            save_format = '{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n'
            for xyxy, track_id in zip(bbox_xyxy, identities):
                if track_id < 0 or track_id is None:
                    continue
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                lines.append(save_format.format(frame=i + 1, id=track_id, x1=x1, y1=y1, w=w, h=h))

                # Draw bounding boxes and IDs on the frame
                if draw_res:
                    color = tuple(self.colors[track_id % 64].tolist())
                    cv2.rectangle(ori_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    text = f"{int(track_id)}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 1)[0]
                    text_x = int(x1) + 3
                    text_y = int(y1) + text_size[1] + 2  # Position inside the top-left corner
                    cv2.putText(ori_img, text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Write the visualization frame to the video
            if draw_res:
                if self.video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(self.save_video_path, fourcc, 10, (seq_w, seq_h))
                self.video_writer.write(cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR))
            
        with open(os.path.join(self.outputs_dir_inference, f'{self.seq_name}.txt'), 'w') as f:
            f.writelines(lines)
        # Release video writer
        if self.video_writer is not None:
            self.video_writer.release()


class RuntimeTrackerBase(object):
    def __init__(self, score_thresh, newborn_thresh, miss_tolerance):
        self.score_thresh = score_thresh
        self.newborn_thresh = newborn_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        device = track_instances.obj_idxes.device

        new_obj = (track_instances.obj_idxes == -1) & (track_instances.scores >= self.newborn_thresh)
        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        disappeared_obj = (track_instances.obj_idxes >= 0) & (track_instances.scores < self.score_thresh)
        num_new_objs = new_obj.sum().item()

        track_instances.obj_idxes[new_obj] = self.max_obj_id + torch.arange(num_new_objs, device=device)
        self.max_obj_id += num_new_objs

        track_instances.disappear_time[disappeared_obj] += 1
        to_del = disappeared_obj & (track_instances.disappear_time >= self.miss_tolerance)
        track_instances.obj_idxes[to_del] = -1


def inference(cfg):
    start_time = time.time()
    # define directory of sequences
    if cfg.mode == 'inference' or cfg.mode == 'train':
        inference_data_dir = os.path.join(cfg.data_root, cfg.inference_dataset, cfg.inference_split)
        seq_nums = os.listdir(inference_data_dir)
        vid_paths = [os.path.join(inference_data_dir, seq) for seq in seq_nums]
    elif cfg.mode == 'video_inference':
        for video in os.listdir(cfg.video_dir):
            assert video.endswith(('.mp4', '.mkv')), f"Make sure all files in {cfg.video_dir} are either .mp4 or .mkv"
        vid_paths = [os.path.join(cfg.video_dir, v) for v in os.listdir(cfg.video_dir) if v.endswith(('.mp4', '.mkv'))]
        
    # load model and weights
    detr, _, _ = build_model(cfg)
    detr.track_embed.score_thr = cfg.score_threshold_inference
    detr.track_base = RuntimeTrackerBase(cfg.score_threshold_inference, cfg.newborn_threshold_inference, cfg.miss_tolerance)
    detr = load_model(detr, cfg.inference_model)
    detr.eval()
    
    if cfg.distributed: # put model on cuda and give each process a subset of all sequences
        detr = detr.to(cfg.gpu)
        vid_paths = [vid_paths[_] for _ in range(len(vid_paths))
                    if _ % get_world_size() == get_rank()]
    else: # just put model on cuda
        detr = detr.cuda()
        
    # create bounding box priors
    if cfg.mode == 'video_inference':
        start_time_bbox_prior = time.time()
        print(f"########### Creating bounding box priors...", flush=True)
        outputs_dir_bbox_priors = os.path.join(cfg.outputs_dir_results, 'bbox_priors')
        for vid_path in vid_paths:
            outputs_path_bbox_priors = os.path.join(outputs_dir_bbox_priors, os.path.basename(vid_path)[:-4] + '.json')
            get_bbox_priors(cfg, vid_path, outputs_path_bbox_priors)
        end_time_bbox_prior = time.time()
        elapsed_time_bbox_prior = end_time_bbox_prior - start_time_bbox_prior
        hours, remainder = divmod(elapsed_time_bbox_prior, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"########### Total time to create bounding box priors: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}", flush=True)
    if cfg.distributed:
        torch.distributed.barrier()
    
    # run inference
    print(f"########### Running inference...", flush=True)
    for vid_path in vid_paths:
        if cfg.mode == 'video_inference':
            cfg.det_db = os.path.join(outputs_dir_bbox_priors, os.path.basename(vid_path)[:-4] + '.json')
        det = Detector(cfg, model=detr, vid_path=vid_path, outputs_dir_results=cfg.outputs_dir_results)
        det.detect(cfg.score_threshold_inference, cfg.area_threshold, draw_res=cfg.visualize_inference)
    if cfg.distributed:
        torch.distributed.barrier()
        
    # Print the total inference time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"########### Total inference time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}", flush=True)
