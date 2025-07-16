# # ------------------------------------------------------------------------
# # Copied from MOTRv2 (https://github.com/megvii-research/MOTRv2)
# # Copyright (c) 2022 megvii-research. All Rights Reserved.
# # ------------------------------------------------------------------------
# # Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# # Copyright (c) 2020 SenseTime. All Rights Reserved.
# # ------------------------------------------------------------------------
# # Modified from DETR (https://github.com/facebookresearch/detr)
# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# # ------------------------------------------------------------------------


import torch
from functools import partial
from models.structures import Instances

def to_cuda(samples, targets, device):
    samples = samples.to(device, non_blocking=True)
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return samples, targets


def tensor_to_cuda(tensor: torch.Tensor, device):
    return tensor.to(device)


def is_tensor_or_instances(data):
    return isinstance(data, torch.Tensor) or isinstance(data, Instances)


def data_apply(data, check_func, apply_func):
    if isinstance(data, dict):
        for k in data.keys():
            if check_func(data[k]):
                data[k] = apply_func(data[k])
            elif isinstance(data[k], dict) or isinstance(data[k], list):
                data_apply(data[k], check_func, apply_func)
            else:
                raise ValueError()
    elif isinstance(data, list):
        for i in range(len(data)):
            if check_func(data[i]):
                data[i] = apply_func(data[i])
            elif isinstance(data[i], dict) or isinstance(data[i], list):
                data_apply(data[i], check_func, apply_func)
            else:
                raise ValueError("invalid type {}".format(type(data[i])))
    else:
        raise ValueError("invalid type {}".format(type(data)))
    return data


def data_dict_to_cuda(data_dict, device):
    return data_apply(data_dict, is_tensor_or_instances, partial(tensor_to_cuda, device=device))
