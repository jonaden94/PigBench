import json
import os
import time
from glob import glob
from pathlib import Path
from mmdet.utils import register_all_modules
from mmdet.apis import init_detector, inference_detector
import mmcv
from tqdm import tqdm
import sys
# make sure script works from wherever it is run
base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)
sys.path.append(os.path.join(base_dir, '../../'))

################### MODYFIABLE ARGS ###################
min_score = 0.5
channel_order = 'rgb'
save_path = f'../../../data/datasets/PigTrack/bbox_priors_minconf{min_score}_{channel_order}.json'

# config and checkpoint file paths
config_path = '../../detector/co_detr/configs/co_dino_swin.py'
checkpoint_path = '../../../data/pretrained/codino/codino_swin.pth'

# build the model from a config and checkpoint
register_all_modules(init_default_scope=False)
model = init_detector(config_path, checkpoint_path, device='cuda:0')

# define image directories of PigDetect
img_dirs_PigDetect = []
base_path = '../../../data/datasets/PigDetect'
allowed_folders = ['dev', 'train', 'val', 'test']
for folder in allowed_folders:
    potential_dir = os.path.join(base_path, folder, 'images')
    if os.path.exists(potential_dir):
        img_dirs_PigDetect.append(potential_dir)

# define image directories of PigTrack
img_dirs_PigTrack = []
base_path = '../../../data/datasets/PigTrack'
allowed_folders = ['dev', 'train', 'val', 'test']
for folder in allowed_folders:
    img_dirs_PigTrack.extend(glob(os.path.join(base_path, folder, '*/img1')))

# instantiate bounding box prior dictionary
bbox_priors = {}

# get time
a = time.time()

# append bounding box prior for PigDetect
print('start creating priors for PigDetect')
for image_dir in tqdm(img_dirs_PigDetect):
    for image_name in os.listdir(image_dir):
        # load image
        image_path = os.path.join(image_dir, image_name)
        cur_img = mmcv.imread(image_path, channel_order=channel_order)

        # run inference
        result = inference_detector(model, cur_img) 
        scores = result.pred_instances.scores.cpu().numpy()
        bboxes = result.pred_instances.bboxes.cpu().numpy()
        
        # filter boxes
        bboxes = bboxes[scores >= min_score]
        scores = scores[scores >= min_score]
        
        detections = []
        for bbox, score in zip(bboxes, scores):
            res = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1], score]
            res = [str(x) for x in res]
            res = ",".join(res) + "\n"
            detections.append(res)

        key = os.path.join('DetDataset', image_name[:-4])
        assert key not in bbox_priors, f"Duplicate key detected: {key}"
        bbox_priors[key] = detections
    
# append bounding box prior for PigTrack
print('start creating priors for PigTrack')
for image_dir in tqdm(img_dirs_PigTrack):
    for image_name in os.listdir(image_dir):
        # load image
        image_path = os.path.join(image_dir, image_name)
        cur_img = mmcv.imread(image_path, channel_order=channel_order)

        # run inference
        result = inference_detector(model, cur_img) 
        scores = result.pred_instances.scores.cpu().numpy()
        bboxes = result.pred_instances.bboxes.cpu().numpy()
        
        # filter boxes
        bboxes = bboxes[scores >= min_score]
        scores = scores[scores >= min_score]
        
        detections = []
        for bbox, score in zip(bboxes, scores):
            res = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1], score]
            res = [str(x) for x in res]
            res = ",".join(res) + "\n"
            detections.append(res)

        split = Path(image_path).parents[2]
        split = os.path.basename(split)
        video_name = Path(image_path).parents[1]
        video_name = os.path.basename(video_name)
        key = os.path.join(video_name, image_name[:-4])   
        assert key not in bbox_priors, f"Duplicate key detected: {key}"
        bbox_priors[key] = detections

with open(save_path, "w") as outfile: 
    json.dump(bbox_priors, outfile)
    
# get time
b = time.time()
# execution time in hours
print(f'execution time: {(b-a)/3600} hours')
print(f'execution time: {b-a} seconds')