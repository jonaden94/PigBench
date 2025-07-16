import os
import argparse

def get_eval_metrics_dict(metric_path: str):
    with open(metric_path) as f:
        metric_names = f.readline()[:-1].split(" ")
        metric_values = f.readline()[:-1].split(" ")
    metrics = {
        n: float(v) for n, v in zip(metric_names, metric_values)
    }
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate tracking metrics")
    parser.add_argument('--gt_dir', required=True, help='Path to ground truth directory')
    parser.add_argument('--tracker_dir', required=True, help='Path to tracker results directory')
    parser.add_argument('--seqmap_path', required=False, help='Path to sequence map file', default=None)
    args = parser.parse_args()

    gt_dir = args.gt_dir
    tracker_dir = args.tracker_dir
    
    if args.seqmap_path is not None:
        data_split = os.path.basename(gt_dir)
        seqmap_path = args.seqmap_path
    else:
        data_split = os.path.basename(gt_dir)
        seqmap_path = os.path.join(os.path.dirname(gt_dir), f'{data_split}_seqmap.txt')

    # path for run_mot_challenge.py
    run_mot_challenge_path = 'TrackEval/scripts/run_mot_challenge.py'

    # eval
    os.system(f"python3 {run_mot_challenge_path} --SPLIT_TO_EVAL {data_split}  "
              f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {gt_dir} "
              f"--SEQMAP_FILE {seqmap_path} "
              f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
              f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False "
              f"--TRACKERS_FOLDER {tracker_dir}")

    eval_metric_path = os.path.join(tracker_dir, "pedestrian_summary.txt")
    eval_metrics_dict = get_eval_metrics_dict(metric_path=eval_metric_path)
    print(eval_metrics_dict)
