import os
import torch
from util.sweeps import log_scores_sweep
from util.misc import is_main_process


def evaluate_inference(cfg):
    trackers_folder = os.path.join(cfg.outputs_dir_results, "tracker")
    inference_dir = os.path.join(cfg.data_root, cfg.inference_dataset, cfg.inference_split)
    
    if is_main_process():
        os.system(f"python3 TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {cfg.inference_split}  "
                    f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {inference_dir} "
                    f"--SEQMAP_FILE {os.path.join(cfg.data_root, cfg.inference_dataset, f'{cfg.inference_split}_seqmap.txt')} "
                    f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
                    f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
                    f"--TRACKERS_FOLDER {trackers_folder} > /dev/null 2>&1")
    if cfg.distributed:
        torch.distributed.barrier()
    
    # Get eval Metrics:
    eval_metric_path = os.path.join(trackers_folder, "pedestrian_summary.txt")
    res_dict = get_eval_metrics_dict(metric_path=eval_metric_path)
    
    if is_main_process():
        log_scores_sweep(HOTA=res_dict['HOTA'], DetA=res_dict['DetA'], AssA=res_dict['AssA'], IDF1=res_dict['IDF1'])
    return res_dict


def get_eval_metrics_dict(metric_path: str):
    with open(metric_path) as f:
        metric_names = f.readline()[:-1].split(" ")
        metric_values = f.readline()[:-1].split(" ")
    metrics = {
        n: float(v) for n, v in zip(metric_names, metric_values)
    }
    return metrics