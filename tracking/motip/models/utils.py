# Copied from MOTIP (https://github.com/MCG-NJU/MOTIP)
# Copyright (c) Ruopeng Gao. All Rights Reserved.

import copy
import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim
from utils.utils import is_distributed, is_main_process
from torch.nn.parallel import DistributedDataParallel as DDP


def get_model(model):
    return model.module if isinstance(model, DDP) else model


def get_activation_layer(activation: str):
    """
    返回一个激活函数层，例如 nn.ReLU。

    Args:
        activation:

    Returns:
    """
    if activation == "ReLU":
        return nn.ReLU(True)
    elif activation == "GELU":
        return nn.GELU()
    elif activation == "SiLU":
        return nn.SiLU(True)
    else:
        raise ValueError(f"Do not support activation layer: {activation}")


def get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


def save_checkpoint(model: nn.Module, path: str, states: dict = None,
                    optimizer: optim = None, scheduler: optim.lr_scheduler = None, only_detr: bool = False):
    if is_main_process():
        model = get_model(model)
        if only_detr:
            model = model.detr
        save_state = {
            "model": model.state_dict(),
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            'states': states
        }
        torch.save(save_state, path)
    else:
        pass
    return


def load_checkpoint(model: nn.Module, path: str, states: dict = None,
                    optimizer: optim = None, scheduler: optim.lr_scheduler = None, logger = None, inference=False):
    load_state = torch.load(path, map_location=lambda storage, loc: storage)
    model_state = load_state["model"]
    target_state_dict = model.state_dict()
    if is_main_process() or inference:
        if "bbox_embed.0.layers.0.weight" in model_state:   # only a detr model from pre-train processing.
            load_detr_pretrain(model=model, pretrain_path=path, num_classes=model.num_classes)
            return  # normally, do not need to load optimizer et al.
        else:
            skip_keys = []
            # skip mismatch size tensors in case of pretraining
            for k in load_state["model"].keys():
                if k not in target_state_dict:
                    continue
                if load_state["model"][k].size() != target_state_dict[k].size():
                    skip_keys.append(k)
            for k in skip_keys:
                del load_state["model"][k]
            missing_keys, unexpected_keys = model.load_state_dict(load_state["model"], strict=False)
            if skip_keys and logger is not None:
                logger.print(f'removed keys in source state_dict due to size mismatch: {", ".join(skip_keys)}')
                logger.save_log_to_file(f'removed keys in source state_dict due to size mismatch: {", ".join(skip_keys)}\n')
            if missing_keys and logger is not None:
                logger.print(f'missing keys in source state_dict: {", ".join(missing_keys)}')
                logger.save_log_to_file(f'missing keys in source state_dict: {", ".join(missing_keys)}\n')
            if unexpected_keys and logger is not None:
                logger.print(f'unexpected key in source state_dict: {", ".join(unexpected_keys)}')
                logger.save_log_to_file(f'unexpected key in source state_dict: {", ".join(unexpected_keys)}\n')
    if optimizer is not None:
        optimizer.load_state_dict(load_state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(load_state["scheduler"])
    if states is not None:
        states.update(load_state["states"])
    return


def load_detr_pretrain(model: nn.Module, pretrain_path: str, num_classes: int):
    pretrain_model = torch.load(pretrain_path, map_location=lambda storage, loc: storage)
    pretrain_state_dict = pretrain_model["model"]
    detr_state_dict = dict()
    model_state_dict = model.state_dict()
    for k, v in pretrain_state_dict.items():
        detr_state_dict["detr."+k] = v
    # 对 Class Head 进行调整，找到匹配的 cls head parameters:
    for k, v in detr_state_dict.items():
        if "class_embed" in k:
            if len(detr_state_dict[k]) == 91:   # Pretrained in coco:
                if num_classes == 1:            # People
                    detr_state_dict[k] = detr_state_dict[k][1:2]
                else:
                    print(">>>> Because the num_classes is not 1, we do not use the pretrained class head.")
                    detr_state_dict[k] = model_state_dict[k]
                    # raise NotImplementedError(f"Do not implement the pretrain loading processing for num_classes={num_classes}")
            elif num_classes == len(detr_state_dict[k]):    # Just fine for the classifier:
                pass
            else:
                raise NotImplementedError(f"Pretrained detr has a class head for {len(detr_state_dict[k])} classes, "
                                          f"we do not support this pretrained model.")
        # For Detect Query:
        if "query_embed" in k:
            if len(detr_state_dict[k]) != len(model_state_dict[k]):
                # missmatch for num of det queries
                print(">>>> Because the num of det queries is not matched, "
                      "we only use a part of the pretrained query embed.")
                detr_state_dict[k] = model_state_dict[k]
            else:
                pass
        # for DINO:
        if "label_enc" in k:
            pass
            if len(detr_state_dict[k]) != len(model_state_dict[k]):
                # missmatch for num classes
                if len(model_state_dict[k]) == 2:   # 1 class
                    detr_state_dict[k] = torch.cat((detr_state_dict[k][1:2], detr_state_dict[k][91:92]), dim=0)
                else:
                    raise NotImplementedError(f"Do not implement the pretrain loading processing for num_classes={num_classes}")
                    pass
    for k, v in detr_state_dict.items():
        assert k in model_state_dict, f"DETR parameter key '{k}' should in the model."
        model_state_dict[k] = v
    model.load_state_dict(state_dict=model_state_dict, strict=True)
    return
