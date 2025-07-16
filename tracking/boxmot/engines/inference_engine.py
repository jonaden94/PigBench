import os
import cv2
import time
import torch
import numpy as np
import gdown
import builtins as __builtin__
from pathlib import Path
from utils.misc import (is_main_process, get_world_size, get_rank)
from utils.datasets import VideoDataset, SeqDataset
from boxmot import create_tracker
from boxmot.appearance.reid_model_factory import get_model_url, __trained_urls
# mmdetection imports
from mmdet.utils import register_all_modules
from mmdet.apis import init_detector, inference_detector


def inference(cfg, tracker_cfg):
    start_time = time.time()
    device = f'cuda:{cfg.gpu}' if cfg.distributed else cfg.device
    
    ################### DETECTOR AND TRACKER
    # detector
    print("[INFO] Initializing detection model...", flush=True)
    register_all_modules(init_default_scope=False)
    detector = init_detector(
        cfg.inference_detector_config,
        cfg.inference_detector_checkpoint,
        device=device
    )
    
    # download reid weights (only on main process in case of distributed inference) and define path to reid weights
    if is_main_process():
        if not (cfg.reid_weights is None or str(cfg.reid_weights).upper() == 'NONE' or str(cfg.reid_weights).upper() == 'FALSE'):
            cfg.reid_weights = Path(cfg.reid_weights)
            assert cfg.reid_weights.suffix == ".pt", "ReID weights must be a .pt file"
            model_url = get_model_url(cfg.reid_weights)
            if model_url is not None:
                os.makedirs('reid_weights', exist_ok=True)
                save_path = os.path.join(f'reid_weights/{str(cfg.reid_weights)}')
                if not os.path.exists(save_path):
                    gdown.download(model_url, save_path, quiet=True)
            else:
                print(f"[INFO] No URL associated with the chosen weights ({str(cfg.reid_weights)}). Available .pt ReID models:", flush=True)
                print(list(__trained_urls.keys()), flush=True)
                exit()
    if cfg.distributed:
        torch.distributed.barrier()
    if cfg.reid_weights is None or str(cfg.reid_weights).upper() == 'NONE' or str(cfg.reid_weights).upper() == 'FALSE':
        reid_weights = None
    else:
        reid_weights = Path(f'reid_weights/{str(cfg.reid_weights)}')
        
    ################### GET SEQUENCES TO PROCESS
    seq_paths = [os.path.join(cfg.seq_dir, _) for _ in os.listdir(cfg.seq_dir)]
    if cfg.distributed:
        seq_paths = [seq_paths[_] for _ in range(len(seq_paths))
                    if _ % get_world_size() == get_rank()]
        
    ################### INFERENCE
    print(f"[INFO] Running tracking inference...", flush=True)
    for seq_path in seq_paths:
        # create separate tracker for every sequence
        tracker = create_tracker(cfg.tracker_type, evolve_param_dict=tracker_cfg, reid_weights=reid_weights, device=device, half=cfg.half)
        # dataset
        if cfg.mode == 'inference':
            seq_name = os.path.basename(seq_path)
            dataset = SeqDataset(seq_path)
        elif cfg.mode == 'video_inference':
            seq_name = os.path.basename(seq_path)[:-4]
            dataset = VideoDataset(seq_path)
            
        # Prepare output files
        colors = (np.random.rand(64, 3) * 255).astype(dtype=np.int32)
        output_file = os.path.join(cfg.tracker_dir, f"{seq_name}.txt")
        save_video_path = os.path.join(cfg.visualization_dir, f"{seq_name}.mp4")
        
        frame_id = 1
        video_writer = None
        lines = []
        for idx in range(len(dataset)):
            frame, frame_ori = dataset[idx]
            seq_h, seq_w, _ = frame_ori.shape

            # Perform detection
            result = inference_detector(detector, frame)
            scores = result.pred_instances.scores.cpu().numpy()
            bboxes = result.pred_instances.bboxes.cpu().numpy()

            # Filter the detections (e.g., based on confidence threshold)
            dets = []
            for i, score in enumerate(scores):
                if score >= cfg.inference_detector_min_conf:
                    bbox = bboxes[i]
                    label = 0
                    conf = score.item()
                    dets.append([*bbox, conf, label])

            # Update the tracker
            dets = np.array(dets) # (N X (x, y, x, y, conf, cls))
            res = tracker.update(dets, frame)  # --> M X (x, y, x, y, id, conf, cls, ind)

            # MOT17 challenge output format
            save_format = '{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1\n'
            for track in res:
                id = int(track[4])
                x1, y1, x2, y2 = track[:4]
                w = x2 - x1
                h = y2 - y1
                conf = track[5]
                line = save_format.format(frame=frame_id, id=id, x1=x1, y1=y1, w=w, h=h, conf=conf)
                lines.append(line)

                # Draw bounding boxes and IDs on the frame
                if cfg.visualize_inference:
                    color = tuple(colors[id % 32].tolist())
                    cv2.rectangle(frame_ori, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    text = f"{int(id)}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 1)[0]
                    text_x = int(x1) + 3
                    text_y = int(y1) + text_size[1] + 2  # Position inside the top-left corner
                    cv2.putText(frame_ori, text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Write the visualization frame to the video
            if cfg.visualize_inference:
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(save_video_path, fourcc, 10, (seq_w, seq_h))
                video_writer.write(cv2.cvtColor(frame_ori, cv2.COLOR_RGB2BGR))
            frame_id += 1
        
        # Release resources
        if cfg.visualize_inference:
            video_writer.release()
        # Write to the file
        with open(output_file, "w") as f:
            f.writelines(lines)
    
    if cfg.distributed:
        torch.distributed.barrier()
    end_time = time.time()
    elapsed_time_bbox_prior = end_time - start_time
    hours, remainder = divmod(elapsed_time_bbox_prior, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"[INFO] Inference completed. Total inference time {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}", flush=True)
