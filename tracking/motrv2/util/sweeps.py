import wandb
from util.misc import is_main_process, is_dist_avail_and_initialized
import os
import yaml
import torch.distributed as dist


# code of these two functions is very hacky (e.g. some path is being hardcoded within these functions)
# not really meant for the public but for personal use for hyperparameter optimization
def init_and_save_sweep(cfg):
    # use custom environment variable to modulate behavior of this function. 
    # Variable must be set to True in the batchscript starting the a sweep.
    if os.environ.get("SWEEP_TIMESTAMP") is not None and is_main_process():
        # Initialize wandb on the main process
        wandb.init()  
        
        # based on sweep identifiers set outputs_base and exp_name and temporary save path of sweep config
        assert wandb.run.sweep_id is not None, "wandb sweep id is None."
        assert wandb.run.id is not None, "wandb run id is None."
        
        # temporarily save the sweep config to a file
        os.makedirs(os.path.join(cfg['outputs_base'], os.environ.get("SWEEP_TIMESTAMP")))
        sweep_config_temp_path = os.path.join(cfg['outputs_base'], os.environ.get("SWEEP_TIMESTAMP"), "sweep_config_temp.yaml")

        # Convert wandb.config to a normal Python dict and save it to a YAML file
        config_dict = dict(wandb.config)
        config_dict["outputs_base"] = os.path.join("outputs", wandb.run.sweep_id)
        config_dict["exp_name"] = wandb.run.id
        with open(sweep_config_temp_path, "w") as f:
            yaml.dump(config_dict, f)
    
    # Wait for all processes to finish initializing wandb
    if is_dist_avail_and_initialized():
        dist.barrier()


def update_config_with_sweep(cfg):
    if os.environ.get("SWEEP_TIMESTAMP"):
        sweep_config_temp_path = os.path.join(cfg['outputs_base'], os.environ.get("SWEEP_TIMESTAMP"), "sweep_config_temp.yaml")
        # Read the YAML file (on all ranks)
        with open(sweep_config_temp_path, "r") as f:
            sweep_hparams = yaml.safe_load(f)
        if is_dist_avail_and_initialized():
            dist.barrier()
            
        if is_main_process():
            os.remove(sweep_config_temp_path)
            os.rmdir(os.path.join(cfg['outputs_base'], os.environ.get("SWEEP_TIMESTAMP")))

        # For each key in the wandb sweep config, assert it’s in cfg and update
        for k, v in sweep_hparams.items():
            if k not in cfg:
                raise KeyError(
                    f"Hyperparameter '{k}' from wandb config is not present in cfg keys.\n"
                    f"cfg keys: {list(cfg.keys())}"
                )
            if k == 'sampler_lengths':
                cfg[k] = v
                if v == [5]:
                    sampler_steps = [0]
                elif v == [5, 7]:
                    sampler_steps = [0, 1]
                elif v == [5, 7, 9]:
                    sampler_steps = [0, 1, 2]
                cfg['sampler_steps'] = sampler_steps
            elif k == 'lr':
                cfg[k] = v
                if v == '0.00004':
                    lr_backbone = 0.000004
                elif v == '0.0002':
                    lr_backbone = 0.00002
                elif v == '0.001':
                    lr_backbone = 0.0001
                cfg['lr_backbone'] = lr_backbone
            elif k == 'epochs':
                lr_drop = k - 3
                cfg['lr_drop'] = lr_drop
            else:
                cfg[k] = v
    
            
def log_scores_sweep(**kwargs):
    if os.environ.get("SWEEP_TIMESTAMP"): 
        for name, value in kwargs.items():
            wandb.log({name: value})
