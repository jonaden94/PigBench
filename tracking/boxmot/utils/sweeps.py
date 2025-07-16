import wandb
from utils.misc import is_main_process, is_dist_avail_and_initialized
import os
import yaml
import wandb
import torch.distributed as dist


# code of these two functions is very hacky (e.g. some path is being hardcoded within these functions)
# not really meant for the public but for personal use for hyperparameter optimization
def init_and_save_sweep(cfg):
    # use custom environment variable to modulate behavior of this function. 
    # Variable must be set to True in the batchscript starting the a sweep.
    if os.environ.get("RUNNING_WANDB_SWEEP") is not None and is_main_process():
        # Initialize wandb on the main process
        wandb.init()  
        
        # save config that contains sweep parameters to adjust base tracker config
        sweep_tracker_config_temp_path = os.path.join(cfg.outputs_base, "sweep_tracker_config_temp.yaml")
        tracker_config_dict = dict(wandb.config)
        with open(sweep_tracker_config_temp_path, "w") as f:
            yaml.dump(tracker_config_dict, f)
            
        # save another config that contains changed arguments for main config
        assert wandb.run.sweep_id is not None, "wandb sweep id is None."
        assert wandb.run.id is not None, "wandb run id is None."
        sweep_config_temp_path = os.path.join(cfg.outputs_base, "sweep_config_temp.yaml")
        config_dict = {
            "outputs_base": os.path.join("outputs", wandb.run.sweep_id),
            "exp_name": wandb.run.id
        }
        with open(sweep_config_temp_path, "w") as f:
            yaml.dump(config_dict, f)
    
    # Wait for all processes to finish initializing wandb
    if is_dist_avail_and_initialized():
        dist.barrier()


def update_config_with_sweep(cfg, config_temp_path):
    if os.environ.get("RUNNING_WANDB_SWEEP"):
        # Read the YAML file (on all ranks)
        with open(config_temp_path, "r") as f:
            sweep_hparams = yaml.safe_load(f)
        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            os.remove(config_temp_path)

        # For each key in the wandb sweep config, assert it’s in cfg and update
        for k, v in sweep_hparams.items():
            if k not in cfg:
                raise KeyError(
                    f"Hyperparameter '{k}' from wandb config is not present in cfg keys.\n"
                    f"cfg keys: {list(cfg.keys())}"
                )
            cfg[k] = v
    
            
def log_scores_sweep(**kwargs):
    if os.environ.get("RUNNING_WANDB_SWEEP"): 
        for name, value in kwargs.items():
            wandb.log({name: value})
