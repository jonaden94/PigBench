# Modified from MOTIP (https://github.com/MCG-NJU/MOTIP)
# Copyright (c) Ruopeng Gao. All Rights Reserved.

import os
import torch
import torch.distributed
from log.log import Metrics
from utils.utils import is_main_process, is_distributed
from utils.sweeps import log_scores_sweep


@torch.no_grad()
def evaluate_inference(config):
    metrics = Metrics()
    inference_split = config["INFERENCE_SPLIT"]
    tracker_dir = os.path.join(config["OUTPUTS_DIR_RESULTS"], "tracker")
    dataset_dir = os.path.join(config["DATA_ROOT"], config["INFERENCE_DATASET"])
    gt_dir = os.path.join(dataset_dir, inference_split)
    inf_split = config['INFERENCE_SPLIT']

    # Need to eval the submit tracker:
    if is_main_process():
        os.system(f"python3 TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {inf_split}  "
                    f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {gt_dir} "
                    f"--SEQMAP_FILE {os.path.join(dataset_dir, f'{inf_split}_seqmap.txt')} "
                    f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
                    f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
                    f"--TRACKERS_FOLDER {tracker_dir}")

    if is_distributed():
        torch.distributed.barrier()
    # Get eval Metrics:
    eval_metric_path = os.path.join(tracker_dir, "pedestrian_summary.txt")
    eval_metrics_dict = get_eval_metrics_dict(metric_path=eval_metric_path)
    metrics["HOTA"].update(eval_metrics_dict["HOTA"])
    metrics["DetA"].update(eval_metrics_dict["DetA"])
    metrics["AssA"].update(eval_metrics_dict["AssA"])
    metrics["DetPr"].update(eval_metrics_dict["DetPr"])
    metrics["DetRe"].update(eval_metrics_dict["DetRe"])
    metrics["AssPr"].update(eval_metrics_dict["AssPr"])
    metrics["AssRe"].update(eval_metrics_dict["AssRe"])
    metrics["MOTA"].update(eval_metrics_dict["MOTA"])
    metrics["IDF1"].update(eval_metrics_dict["IDF1"])
    
    if is_main_process():
        log_scores_sweep(HOTA=eval_metrics_dict['HOTA'], DetA=eval_metrics_dict['DetA'], AssA=eval_metrics_dict['AssA'], IDF1=eval_metrics_dict['IDF1'])
    return metrics


def get_eval_metrics_dict(metric_path: str):
    with open(metric_path) as f:
        metric_names = f.readline()[:-1].split(" ")
        metric_values = f.readline()[:-1].split(" ")
    metrics = {
        n: float(v) for n, v in zip(metric_names, metric_values)
    }
    return metrics
