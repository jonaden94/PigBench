# ------------------------------------------------------------------------
# Modified from MOTRv2 (https://github.com/megvii-research/MOTRv2)
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import pprint
import random
import os
import sys
import numpy as np
import torch
import torch.distributed
from util.misc import (init_distributed_mode, get_rank, is_main_process, 
                       munch_to_dict, write_dict_to_yaml)
from util.parser import yaml_to_dict, update_config, parse_option, load_super_config
from util.sweeps import init_and_save_sweep, update_config_with_sweep
from engines.train_engine import train
from engines.inference_engine import inference
from engines.eval_engine import evaluate_inference


def main(cfg):  
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_file_dir) 
    init_distributed_mode(cfg)
    
    # if wandb sweep is being run: init wandb and change config with sweep values
    # also cfg.outputs_base and cfg.exp_name are set based on sweep identifiers
    init_and_save_sweep(cfg)
    update_config_with_sweep(cfg)
    
    # directory for saving outputs
    cfg.outputs_dir = os.path.join(cfg.outputs_base, cfg.exp_name)
    if cfg.mode == "train":
        log_dir = os.path.join(cfg.outputs_dir, cfg.mode)
        cfg.outputs_dir_results_base = os.path.join(log_dir, 'eval_during_train', cfg.inference_split)
    elif cfg.mode == "inference":
        log_dir = os.path.join(cfg.outputs_dir, cfg.mode, cfg.inference_split, cfg.inference_model.split("/")[-1][:-4])
        cfg.outputs_dir_results = os.path.join(log_dir, 'results')
    elif cfg.mode == "video_inference":
        log_dir = os.path.join(cfg.outputs_dir, cfg.mode, cfg.video_dir.split("/")[-1], cfg.inference_model.split("/")[-1][:-4])
        cfg.outputs_dir_results = os.path.join(log_dir, 'results')
    else:
        raise NotImplementedError(f"Do not support running mode '{cfg.mode}'.")
    
    # init logging and save config
    if is_main_process():
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'log.txt')
        sys.stdout = open(log_file, "w")
        sys.stderr = sys.stdout
        write_dict_to_yaml(cfg, os.path.join(log_dir, 'config.yaml'))
    if cfg.distributed:
        torch.distributed.barrier()
     
     # print config   
    formatted_cfg = pprint.pformat(munch_to_dict(cfg), indent=2)
    print(formatted_cfg, flush=True)

    # fix the seed for reproducibility
    seed = cfg.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if cfg.mode == 'train':
        train(cfg)
    elif cfg.mode == 'inference' or cfg.mode == 'video_inference':
        inference(cfg)
        if cfg.evaluate_inference_mode:
            evaluate_inference(cfg)
        
    else:
        raise NotImplementedError(f"Do not support running mode '{cfg.mode}'.")


if __name__ == '__main__':
    opt = parse_option()
    cfg = yaml_to_dict(opt.config)
    cfg = load_super_config(cfg, cfg["super_config_path"])
    cfg = update_config(cfg, opt)
    main(cfg)
