import os
import sys
import yaml
import torch
import builtins as __builtin__
from utils.misc import (init_distributed_mode, yaml_to_dict, update_config, 
                         load_super_config, write_dict_to_yaml,
                         is_main_process, parse_option, get_tracker_config_path)
from engines.inference_engine import inference
from engines.eval_engine import evaluate_inference
from utils.sweeps import init_and_save_sweep, update_config_with_sweep


def main(cfg):
    init_distributed_mode(cfg)
    init_and_save_sweep(cfg)
    # get tracker config and update them in case of hyperparameter sweep
    tracker_config_path = get_tracker_config_path(cfg.tracker_type)
    with open(tracker_config_path, "r") as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)
    tracker_cfg = {param: details['default'] for param, details in yaml_config.items()}
    update_config_with_sweep(tracker_cfg, os.path.join(cfg.outputs_base, "sweep_tracker_config_temp.yaml"))
    update_config_with_sweep(cfg, os.path.join(cfg.outputs_base, "sweep_config_temp.yaml"))
    
    ################### OUTPUT
    name = os.path.basename(cfg.seq_dir.rstrip('/'))
    cfg.outputs_dir = os.path.join(cfg.outputs_base, cfg.exp_name, 'inference', name)
    results_dir = os.path.join(cfg.outputs_dir, 'results')
    cfg.tracker_dir = os.path.join(results_dir, 'tracker')
    cfg.visualization_dir = os.path.join(results_dir, 'visualization')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(cfg.tracker_dir, exist_ok=True)
    os.makedirs(cfg.visualization_dir, exist_ok=True)

    ################### DOCUMENTATION
    # Write config to a YAML file in the output directory
    if is_main_process():
        # save configs
        config_save_path = os.path.join(cfg.outputs_dir, 'config.yaml')
        write_dict_to_yaml(cfg, config_save_path)
        tracker_config_save_path = os.path.join(cfg.outputs_dir, 'tracker_config.yaml')
        write_dict_to_yaml(tracker_cfg, tracker_config_save_path)
        # Log file capture
        log_file = os.path.join(cfg.outputs_dir, 'log.txt')
        sys.stdout = open(log_file, "w")
        sys.stderr = sys.stdout
    if cfg.distributed:
        torch.distributed.barrier()

    # run inference and optionally evaluation        
    inference(cfg, tracker_cfg)
    if cfg.evaluate_inference_mode:
        evaluate_inference(cfg)
        

if __name__ == '__main__':
    opt = parse_option()
    cfg = yaml_to_dict(opt.config)
    cfg = load_super_config(cfg, cfg["super_config_path"])
    cfg = update_config(cfg, opt)
    main(cfg)
