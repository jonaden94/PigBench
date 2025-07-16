# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : This file will update the config.

import argparse
from utils.utils import yaml_to_dict


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
            config_k = option_k.upper()
            if config_k in config:
                if option_v == "True":
                    config[config_k] = True
                elif option_v == "False":
                    config[config_k] = False
                elif isinstance(option_v, str) and "," in option_v:
                    config[config_k] = [item.strip() for item in option_v.split(",") if item.strip()]
                    config[config_k] = [None if item == "None" or item == '~' else item for item in config[config_k]]
                elif isinstance(option_v, str) and (option_v == 'None' or option_v == '~'):
                    config[config_k] = None
                else:
                    config[config_k] = option_v
            else:
                raise RuntimeError(f"The option '{option_k}' is not appeared in .yaml config file.")
    return config


def load_super_config(config: dict, super_config_path: str | None):
    if super_config_path is None:
        return config
    else:
        super_config = yaml_to_dict(super_config_path)
        super_config = load_super_config(super_config, super_config["SUPER_CONFIG_PATH"])
        super_config.update(config)
        return super_config
