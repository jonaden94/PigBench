import os
import yaml
import torch
import argparse
from munch import Munch
import subprocess
import torch.distributed as dist
import builtins as __builtin__
import warnings

############################################# DISTRIBUTED
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def is_distributed():
    if not (dist.is_available() and dist.is_initialized()):
        return False
    return True

def setup_for_distributed(is_master):
    """
    This function disables printing and warnings when not in the master process.
    """

    # Store the original print function
    builtin_print = __builtin__.print

    # Define a custom print function
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    # Override the built-in print function
    __builtin__.print = print

    # Suppress warnings for non-master processes
    if not is_master:
        warnings.filterwarnings("ignore")

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    
    torch.cuda.set_device(args.gpu)

    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    dist.barrier()
    setup_for_distributed(args.rank == 0)
    
############################################# PARSER
def parse_option():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--config', type=str,
                        help='Path to the primary YAML config file.')

    parser.add_argument('--mode', type=str,
                        help='Execution mode: "inference", or "video_inference".')
    parser.add_argument('--seq_dir', type=str,
                        help='Path to directory containing image sequences or video files.')
    parser.add_argument('--exp_name', type=str,
                        help='Experiment name for organizing outputs.')
    parser.add_argument('--outputs_base', type=str,
                        help='Base directory for saving outputs.')
    parser.add_argument('--visualize_inference', type=str, 
                        help='Whether to visualize the inference results.') 
    parser.add_argument('--evaluate_inference_mode', type=str, 
                        help='Whether to evaluate the inference results.')
    parser.add_argument('--gt_dir', type=str,
                        help='Path to directory containing ground truth annotations used for evaluation')

    # Tracker Settings
    parser.add_argument('--tracker_type', type=str,
                        help='Type of tracker to use (e.g., "botsort", "strongsort").')
    parser.add_argument('--reid_weights', type=str,
                        help='Path to ReID model weights for the tracker.')
    parser.add_argument('--half', action='store_true',
                        help='Use half precision (FP16) for the tracker if supported.')

    # Detector Settings
    parser.add_argument('--inference_detector_config', type=str,
                        help='Path to the detector config file.')
    parser.add_argument('--inference_detector_checkpoint', type=str,
                        help='Path to the detector checkpoint.')
    parser.add_argument('--inference_detector_min_conf', type=float,
                        help='Minimum confidence threshold for valid detections.')

    # Other Settings
    parser.add_argument('--device', type=str,
                        help='Device to use if not running distributed (e.g., "cpu", "cuda").')
    return parser.parse_args()

def write_dict_to_yaml(x: dict, savepath: str, mode: str = "w"):
    # Convert Munch back to dict if needed
    if isinstance(x, Munch):
        x = x.toDict()
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    with open(savepath, mode) as f:
        yaml.dump(x, f, allow_unicode=True)
    return

def yaml_to_dict(path: str):
    with open(path) as f:
        return Munch.fromDict(yaml.load(f.read(), yaml.FullLoader))
    
def update_config(config: dict, option: argparse.Namespace) -> dict:
    """
    Update current config with an option parser.

    Args:
        config: Current config.
        option: Option parser.

    Returns:
        New config dict.
    """
    for option_k, option_v in vars(option).items():
        if option_k != "config" and option_v is not None:
            if option_k in config:
                if str(option_v).upper() == "TRUE":
                    config[option_k] = True
                elif str(option_v).upper() == "FALSE":
                    config[option_k] = False
                else:
                    config[option_k] = option_v
            else:
                raise RuntimeError(f"The option '{option_k}' does not appear in .yaml config file.")
    return config

def load_super_config(config: dict, super_config_path: str | None):
    if super_config_path is None:
        return config
    else:
        super_config = yaml_to_dict(super_config_path)
        super_config = load_super_config(super_config, super_config["super_config_path"])
        super_config.update(config)
        return super_config

############################################# MODEL
def get_tracker_config_path(tracker_type: str) -> str:
    if tracker_type == 'bytetrack':
        return 'configs/trackers/bytetrack.yaml'
    elif tracker_type == 'botsort':
        return 'configs/trackers/botsort.yaml'
    elif tracker_type == 'strongsort':
        return 'configs/trackers/strongsort.yaml'
    elif tracker_type == 'ocsort':
        return 'configs/trackers/ocsort.yaml'
    elif tracker_type == 'deepocsort':
        return 'configs/trackers/deepocsort.yaml'
    elif tracker_type == 'hybridsort':
        return 'configs/trackers/hybridsort.yaml'
    elif tracker_type == 'imprassoc':
        return 'configs/trackers/imprassoc.yaml'
    else:
        raise ValueError(f"Tracker type '{tracker_type}' not supported.")
