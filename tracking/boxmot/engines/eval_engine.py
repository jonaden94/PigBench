import os
import torch
from utils.misc import is_main_process, is_distributed
from utils.sweeps import log_scores_sweep


def evaluate_inference(cfg):
    # pred data
    # ground truth data
    if is_main_process():
        inference_dir = cfg.gt_dir
        inference_base = os.path.dirname(inference_dir)
        inference_split = os.path.basename(inference_dir)
        seqmap_path = os.path.join(inference_base, f'{inference_split}_seqmap.txt')
        os.system(f"python3 TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {inference_split}  "
                    f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {inference_dir} "
                    f"--SEQMAP_FILE {seqmap_path} "
                    f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
                    f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
                    f"--TRACKERS_FOLDER {cfg.tracker_dir} > /dev/null 2>&1")
    if is_distributed():
        torch.distributed.barrier()
    
    # Get eval Metrics:
    eval_metric_path = os.path.join(cfg.tracker_dir, "pedestrian_summary.txt")
    eval_metrics_dict = get_eval_metrics_dict(metric_path=eval_metric_path)
    print(f'[INFO] Finished evaluation: HOTA: {eval_metrics_dict["HOTA"]}, DetA: {eval_metrics_dict["DetA"]}, AssA: {eval_metrics_dict["AssA"]}, IDF1: {eval_metrics_dict["IDF1"]}')
    if is_main_process():
        log_scores_sweep(HOTA=eval_metrics_dict['HOTA'], DetA=eval_metrics_dict['DetA'], AssA=eval_metrics_dict['AssA'], IDF1=eval_metrics_dict['IDF1'])


def get_eval_metrics_dict(metric_path: str):
    with open(metric_path) as f:
        metric_names = f.readline()[:-1].split(" ")
        metric_values = f.readline()[:-1].split(" ")
    metrics = {
        n: float(v) for n, v in zip(metric_names, metric_values)
    }
    return metrics
