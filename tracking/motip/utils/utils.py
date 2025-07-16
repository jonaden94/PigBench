# Copied from MOTIP (https://github.com/MCG-NJU/MOTIP)
# Copyright (c) Ruopeng Gao. All Rights Reserved.

import os
import math
import random
import argparse
import yaml
import torch
import torchvision
import torch.distributed
import random
import subprocess
import numpy as np
from munch import Munch
from typing import Any, Generator, List


def parse_option():
    """
    Build a parser that can set up runtime options, such as choose device, data path, and so on.
    Every option in this parser should appear in .yaml config file (like ./configs/resnet18_mnist.yaml),
    except --config.

    Returns:
        A parser.

    """
    parser = argparse.ArgumentParser("Network training and evaluation script.", add_help=True)

    # Git:
    parser.add_argument("--git-version", type=str)

    # Running mode, Training? Evaluation? or ?
    parser.add_argument("--mode", type=str, help="Running mode.")
    # logging every n steps
    parser.add_argument("--outputs-per-step", type=int, help="Print log every n steps.")
    parser.add_argument("--outputs-base", type=str, help="Output directory.")

    # About model setting:
    parser.add_argument("--backbone", type=str)
    parser.add_argument("--detr-num-queries", type=int)
    parser.add_argument("--pretrain", type=str) # full model pretrain
    parser.add_argument("--detr-pretrain", type=str)    # DETR pretrain
    parser.add_argument("--seq-hidden-dim", type=int)
    parser.add_argument("--seq-dim-feedforward", type=int)
    parser.add_argument("--id-decoder-layers", type=int)
    parser.add_argument("--num-id-vocabulary", type=int)

    # Config file.
    parser.add_argument("--config", type=str, help="Config file path.",
                        default="./configs/resnet18_mnist.yaml")
    parser.add_argument("--super-config-path", type=str)

    # About system.
    parser.add_argument("--device", type=str, help="Device.")
    parser.add_argument("--num-cpu-per-gpu", type=int)
    parser.add_argument("--num-workers", type=int)

    # About data.
    parser.add_argument("--data-path", type=str, help="Data path.")
    parser.add_argument("--data-root", type=str)
    parser.add_argument("--sample-lengths", type=int)
    parser.add_argument("--sample-intervals", type=int)
    parser.add_argument("--max-temporal-length", type=int)
    parser.add_argument("--aug-random-shift-max-ratio", type=float)
    parser.add_argument("--dataset-weights", type=float)
    parser.add_argument("--datasets", type=str)
    parser.add_argument("--dataset-splits", type=str)
    parser.add_argument("--dataset-seqmap-names", type=str)
    parser.add_argument("--dataset-types", type=str)

    # About outputs.
    parser.add_argument("--use-wandb", type=str)
    parser.add_argument("--exp-owner", type=str)
    parser.add_argument("--exp-name", type=str, help="Exp name.")
    parser.add_argument("--exp-group", type=str, help="Exp group, for wandb.")
    parser.add_argument("--save-checkpoint-per-epoch", type=int)

    # About train setting:
    parser.add_argument("--resume-model", type=str, help="Resume training model path.")
    parser.add_argument("--resume-optimizer", type=str)
    parser.add_argument("--resume-scheduler", type=str)
    parser.add_argument("--detr-num-train-frames", type=int)
    parser.add_argument("--accumulate-steps", type=int)
    parser.add_argument("--detr-checkpoint-frames", type=int)
    parser.add_argument("--seq-decoder-checkpoint", type=str)
    parser.add_argument("--training-num-id", type=int)
    parser.add_argument("--memory-optimized-detr-criterion", type=str)
    parser.add_argument("--auto-memory-optimized-detr-criterion", type=str)
    parser.add_argument("--checkpoint-detr-criterion", type=str)
    # Training augmentation parameters:
    parser.add_argument("--traj-drop-ratio", type=float)
    parser.add_argument("--traj-switch-ratio", type=float)

    # About evaluation and submit:
    parser.add_argument("--inference-model", type=str)
    parser.add_argument("--inference-config-path", type=str)
    parser.add_argument("--inference-group", type=str)
    parser.add_argument("--inference-split", type=str)
    parser.add_argument("--inference-dataset", type=str)
    parser.add_argument("--inference-max-size", type=int)
    parser.add_argument("--evaluate-inference-mode", type=str)

    # Hyperparams.
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--scheduler-milestones", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr-warmup-epochs", type=int)
    parser.add_argument("--id-thresh", type=float)
    parser.add_argument("--det-thresh", type=float)
    parser.add_argument("--newborn-thresh", type=float)
    parser.add_argument("--area-thresh", type=int)
    parser.add_argument("--detr-cls-loss-coef", type=float)
    
    # video infer
    parser.add_argument("--video-dir", type=str)
    parser.add_argument("--visualize-inference", type=str)
    parser.add_argument("--use-positional-encoding", type=str)
    parser.add_argument("--push-forward-thresh", type=int)
    parser.add_argument("--patience-single", type=int)
    parser.add_argument("--patience-multiple", type=int)

    # dist without slurm
    parser.add_argument("--local-rank", type=int)

    return parser.parse_args()


def set_seed(seed: int):
    seed = seed + distributed_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return


def is_distributed():
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return False
    return True


def distributed_rank():
    if not is_distributed():
        return 0
    else:
        return torch.distributed.get_rank()


def distributed_world_rank():
    if not is_distributed():
        return 0
    else:
        return torch.distributed.get_global_rank()


def is_main_process():
    return distributed_rank() == 0 or (
        'SLURM_PROCID' in os.environ and int(os.environ['SLURM_PROCID']) == 0
    )


def distributed_world_size():
    if is_distributed():
        return torch.distributed.get_world_size()
    else:
        return 1


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
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
    
    
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def yaml_to_dict(path: str):
    """
    Read a yaml file into a dict.

    Args:
        path (str): The path of yaml file.

    Returns:
        A dict.
    """
    with open(path) as f:
        return Munch.fromDict(yaml.load(f.read(), yaml.FullLoader))
    
    
def munch_to_dict(obj):
    if isinstance(obj, Munch):
        return {key: munch_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [munch_to_dict(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(munch_to_dict(item) for item in obj)
    else:
        return obj


def labels_to_one_hot(labels: torch.Tensor, class_num: int, device="cpu"):
    """
    Args:
        labels: Original labels.
        class_num:
        device:
    Returns:
        Labels in one-hot.
    """
    if len(labels) > 0:
        return torch.eye(n=class_num, device=device)[labels].reshape((len(labels), -1))
    else:
        # A hack for empty labels.
        return torch.empty((0, class_num), device=device)


def pos_to_pos_embed(pos, num_pos_feats: int = 64, temperature: int = 10000, scale: float = 2 * math.pi):
    """
    Args:
        pos: 0~1, position vector, (N, M) / (B, N, M)
        num_pos_feats:
        temperature:
        scale:

    Returns:
    """
    pos = pos * scale
    dim_i = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_i = temperature ** (2 * (torch.div(dim_i, 2, rounding_mode="trunc")) / num_pos_feats)
    # 10000^(2i/d_model)
    pos_embed = pos[..., None] / dim_i      # (N, M, n_feats) or (B, N, M, n_feats)
    pos_embed = torch.stack((pos_embed[..., 0::2].sin(), pos_embed[..., 1::2].cos()), dim=-1)
    pos_embed = torch.flatten(pos_embed, start_dim=-3)
    return pos_embed


def inverse_sigmoid(x, eps=1e-5):
    """
    if      x = 1/(1+exp(-y))
    then    y = ln(x/(1-x))
    Args:
        x:
        eps:

    Returns:

    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    """
    return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def combine_detr_outputs(detr_outputs1, detr_outputs2):
    if detr_outputs1 is None:
        return detr_outputs2
    if detr_outputs2 is None:
        return detr_outputs1
    combined_outputs = dict()
    combined_outputs["pred_logits"] = torch.cat([detr_outputs1["pred_logits"], detr_outputs2["pred_logits"]], dim=0)
    combined_outputs["pred_boxes"] = torch.cat([detr_outputs1["pred_boxes"], detr_outputs2["pred_boxes"]], dim=0)
    combined_outputs["outputs"] = torch.cat([detr_outputs1["outputs"], detr_outputs2["outputs"]], dim=0)
    combined_outputs["aux_outputs"] = [
        {
            "pred_logits": torch.cat([
                detr_outputs1["aux_outputs"][_]["pred_logits"],
                detr_outputs2["aux_outputs"][_]["pred_logits"]],
                dim=0
            ),
            "pred_boxes": torch.cat([
                detr_outputs1["aux_outputs"][_]["pred_boxes"],
                detr_outputs2["aux_outputs"][_]["pred_boxes"]],
                dim=0
            ),
        }
        for _ in range(len(detr_outputs1["aux_outputs"]))
    ]
    if "dn_meta" in detr_outputs1:  # for DINO?
        combined_outputs["dn_meta"] = {}
        combined_outputs["dn_meta"]["pad_size"] = detr_outputs1["dn_meta"]["pad_size"]
        combined_outputs["dn_meta"]["num_dn_group"] = detr_outputs1["dn_meta"]["num_dn_group"]
        combined_outputs["dn_meta"]["output_known_lbs_bboxes"] = {}
        combined_outputs["dn_meta"]["output_known_lbs_bboxes"]["pred_logits"] = torch.cat([
            detr_outputs1["dn_meta"]["output_known_lbs_bboxes"]["pred_logits"],
            detr_outputs2["dn_meta"]["output_known_lbs_bboxes"]["pred_logits"]],
            dim=0
        )
        combined_outputs["dn_meta"]["output_known_lbs_bboxes"]["pred_boxes"] = torch.cat([
            detr_outputs1["dn_meta"]["output_known_lbs_bboxes"]["pred_boxes"],
            detr_outputs2["dn_meta"]["output_known_lbs_bboxes"]["pred_boxes"]],
            dim=0
        )
        combined_outputs["dn_meta"]["output_known_lbs_bboxes"]["aux_outputs"] = [
            {
                "pred_logits": torch.cat([
                    detr_outputs1["dn_meta"]["output_known_lbs_bboxes"]["aux_outputs"][_]["pred_logits"],
                    detr_outputs2["dn_meta"]["output_known_lbs_bboxes"]["aux_outputs"][_]["pred_logits"]],
                    dim=0
                ),
                "pred_boxes": torch.cat([
                    detr_outputs1["dn_meta"]["output_known_lbs_bboxes"]["aux_outputs"][_]["pred_boxes"],
                    detr_outputs2["dn_meta"]["output_known_lbs_bboxes"]["aux_outputs"][_]["pred_boxes"]],
                    dim=0
                ),
            }
            for _ in range(len(detr_outputs1["dn_meta"]["output_known_lbs_bboxes"]["aux_outputs"]))
        ]
    if "interm_outputs" in detr_outputs1:
        combined_outputs["interm_outputs"] = {
            "pred_logits": torch.cat(
                [detr_outputs1["interm_outputs"]["pred_logits"], detr_outputs2["interm_outputs"]["pred_logits"]], dim=0
            ),
            "pred_boxes": torch.cat(
                [detr_outputs1["interm_outputs"]["pred_boxes"], detr_outputs2["interm_outputs"]["pred_boxes"]], dim=0
            )
        }
        combined_outputs["interm_outputs_for_matching_pre"] = {
            "pred_logits": torch.cat(
                [detr_outputs1["interm_outputs_for_matching_pre"]["pred_logits"], detr_outputs2["interm_outputs_for_matching_pre"]["pred_logits"]], dim=0
            ),
            "pred_boxes": torch.cat(
                [detr_outputs1["interm_outputs_for_matching_pre"]["pred_boxes"], detr_outputs2["interm_outputs_for_matching_pre"]["pred_boxes"]], dim=0
            )
        }
    return combined_outputs


def detr_outputs_index_select(detr_outputs, index, dim: int = 0):
    selected_detr_outputs = dict()
    selected_detr_outputs["pred_logits"] = torch.index_select(detr_outputs["pred_logits"], index=index, dim=dim).contiguous()
    selected_detr_outputs["pred_boxes"] = torch.index_select(detr_outputs["pred_boxes"], index=index, dim=dim).contiguous()
    selected_detr_outputs["outputs"] = torch.index_select(detr_outputs["outputs"], index=index, dim=dim).contiguous()
    selected_detr_outputs["aux_outputs"] = [
        {
            "pred_logits": torch.index_select(detr_outputs["aux_outputs"][_]["pred_logits"], index=index, dim=dim).contiguous(),
            "pred_boxes": torch.index_select(detr_outputs["aux_outputs"][_]["pred_boxes"], index=index, dim=dim).contiguous(),
        }
        for _ in range(len(detr_outputs["aux_outputs"]))
    ]
    if "dn_meta" in detr_outputs:
        selected_detr_outputs["dn_meta"] = {
            "pad_size": detr_outputs["dn_meta"]["pad_size"],
            "num_dn_group": detr_outputs["dn_meta"]["num_dn_group"],
            "output_known_lbs_bboxes": {
                "pred_logits": torch.index_select(detr_outputs["dn_meta"]["output_known_lbs_bboxes"]["pred_logits"], index=index, dim=dim).contiguous(),
                "pred_boxes": torch.index_select(detr_outputs["dn_meta"]["output_known_lbs_bboxes"]["pred_boxes"], index=index, dim=dim).contiguous(),
                "aux_outputs": [
                    {
                        "pred_logits": torch.index_select(detr_outputs["dn_meta"]["output_known_lbs_bboxes"]["aux_outputs"][_]["pred_logits"], index=index, dim=dim).contiguous(),
                        "pred_boxes": torch.index_select(detr_outputs["dn_meta"]["output_known_lbs_bboxes"]["aux_outputs"][_]["pred_boxes"], index=index, dim=dim).contiguous()
                    }
                    for _ in range(len(detr_outputs["dn_meta"]["output_known_lbs_bboxes"]["aux_outputs"]))
                ],
            }
        }
        pass
    if "interm_outputs" in detr_outputs:
        selected_detr_outputs["interm_outputs"] = {
            "pred_logits": torch.index_select(detr_outputs["interm_outputs"]["pred_logits"], index=index, dim=dim).contiguous(),
            "pred_boxes": torch.index_select(detr_outputs["interm_outputs"]["pred_boxes"], index=index, dim=dim).contiguous()
        }
        selected_detr_outputs["interm_outputs_for_matching_pre"] = {
            "pred_logits": torch.index_select(detr_outputs["interm_outputs_for_matching_pre"]["pred_logits"], index=index, dim=dim).contiguous(),
            "pred_boxes": torch.index_select(detr_outputs["interm_outputs_for_matching_pre"]["pred_boxes"], index=index, dim=dim).contiguous()
        }
    return selected_detr_outputs


def infos_to_detr_targets(infos: dict, device):
    targets = list()
    for info in infos:
        for _ in range(len(info)):
            targets.append({
                "boxes": info[_]["boxes"].to(device),
                "labels": info[_]["labels"].to(device)
            })
    return targets


def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size: (b + 1) * batch_size] for arg in args]


if __name__ == '__main__':
    config = yaml_to_dict("../configs/resnet18_mnist.yaml")


