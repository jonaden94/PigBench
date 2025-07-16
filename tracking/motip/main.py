# Modified from MOTIP (https://github.com/MCG-NJU/MOTIP)
# Copyright (c) Ruopeng Gao. All Rights Reserved.

import os
import torch
from utils.utils import (yaml_to_dict, is_main_process, set_seed,
                         init_distributed_mode, parse_option, munch_to_dict)
from log.logger import Logger
from configs.utils import update_config, load_super_config
from utils.sweeps import init_and_save_sweep, update_config_with_sweep
from engines.train_engine import train
from engines.inference_engine import inference
from engines.eval_engine import evaluate_inference
import pprint


def main(config: dict):
    """
    Main function.

    Args:
        config: Model configs.
    """
    # init distributed mode depending on environment variables
    init_distributed_mode(config)
    
    # if wandb sweep is being run: init wandb and change config with sweep values
    # also cfg.outputs_base and cfg.exp_name are set based on sweep identifiers
    init_and_save_sweep(config)
    update_config_with_sweep(config)

    # set directory where to save outputs
    config["OUTPUTS_DIR"] = os.path.join(config['OUTPUTS_BASE'], config["EXP_NAME"])
    if config["MODE"] == "train":
        log_dir = os.path.join(config["OUTPUTS_DIR"], config["MODE"])
        config['OUTPUTS_DIR_RESULTS_BASE'] = os.path.join(log_dir, 'eval_during_train', config["INFERENCE_SPLIT"])
    elif config["MODE"] == "inference":
        log_dir = os.path.join(config["OUTPUTS_DIR"], config["MODE"], config["INFERENCE_SPLIT"], config["INFERENCE_MODEL"].split("/")[-1][:-4])
        config["OUTPUTS_DIR_RESULTS"] = os.path.join(log_dir, 'results')
    elif config["MODE"] == "video_inference":
        log_dir = os.path.join(config["OUTPUTS_DIR"], config["MODE"], config["VIDEO_DIR"].split("/")[-1], config["INFERENCE_MODEL"].split("/")[-1][:-4])
        config["OUTPUTS_DIR_RESULTS"] = os.path.join(log_dir, 'results')
    else:
        raise NotImplementedError(f"Do not support running mode '{config['MODE']}'.")

    logger = Logger(
        logdir=log_dir,
        use_tensorboard=config["USE_TENSORBOARD"],
        use_wandb=config["USE_WANDB"],
        only_main=True,
        config=config
    )
    # Log runtime config.
    if is_main_process():
        logger.print_config(config=config, prompt="Runtime Configs: ")
        logger.save_config(config=config, filename="config.yaml")
        logger.save_log_to_file(pprint.pformat(munch_to_dict(config)) + '\n\n', mode='w')

    # set seed
    set_seed(config["SEED"])
    # Set num of CPUs
    if "NUM_CPU_PER_GPU" in config and config["NUM_CPU_PER_GPU"] is not None:
        torch.set_num_threads(config["NUM_CPU_PER_GPU"])

    if config["MODE"] == "train":
        train(config=config, logger=logger)
    elif config["MODE"] == "inference" or config["MODE"] == "video_inference":
        inference(config=config, logger=logger)
        if config["EVALUATE_INFERENCE_MODE"]:
            evaluate_inference(config)
    else:
        raise NotImplementedError(f"Do not support running mode '{config['MODE']}'.")
    return


if __name__ == '__main__':
    opt = parse_option()
    cfg = yaml_to_dict(opt.config)
    cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])
    cfg = update_config(config=cfg, option=opt)
    main(config=cfg)
