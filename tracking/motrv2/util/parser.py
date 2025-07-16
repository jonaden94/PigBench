import yaml
import argparse
from munch import Munch


def parse_option():
    parser = argparse.ArgumentParser('motrv2', add_help=False)
    parser.add_argument('--print_freq', type=float, help='print training progress every print_freq iterations')
    parser.add_argument('--mode', type=str, help='train, inference or video_inference')
    parser.add_argument('--config', type=str, help="Config file path")
    parser.add_argument('--outputs_base', type=str, help="Base directory for saving outputs")
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_backbone_names', type=str, nargs='+')
    parser.add_argument('--lr_backbone', type=float)
    parser.add_argument('--lr_linear_proj_names', type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr_drop', type=int)
    parser.add_argument('--clip_max_norm', type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', type=str, help='provide true if using option and false if not')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', type=str, help='provide true if using option and false if not')
    parser.add_argument('--two_stage', type=str, help='provide true if using option and false if not')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--num_feature_levels', type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', type=int)
    parser.add_argument('--enc_n_points', type=int)
    parser.add_argument('--decoder_cross_self', type=str, help='provide true if using option and false if not')
    parser.add_argument('--sigmoid_attn', type=str, help='provide true if using option and false if not')
    parser.add_argument('--extra_track_attn', type=str, help='provide true if using option and false if not')

    # * Segmentation
    parser.add_argument('--masks', type=str, help='provide true if using option and false if not')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # Matcher
    parser.add_argument('--set_cost_class', type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', type=float)
    parser.add_argument('--bbox_loss_coef', type=float)
    parser.add_argument('--giou_loss_coef', type=float)

    # others
    parser.add_argument('--output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device',
                        help='device to use for training / testing')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resume', help='resume training from given checkpoint/epoch')
    parser.add_argument('--start_epoch', type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=str, help='provide true if using option and false if not')
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--pretrained', help='start training from given checkpoint')
    parser.add_argument('--cache_mode', type=str, help='provide true if using option and false if not') # whether to cache images on memory

    # end-to-end mot settings.
    parser.add_argument('--det_db', type=str)
    parser.add_argument('--query_interaction_layer', type=str,
                        help="")
    parser.add_argument('--sample_mode', type=str)
    parser.add_argument('--sample_interval', type=int)
    parser.add_argument('--random_drop', type=float)
    parser.add_argument('--fp_ratio', type=float)
    parser.add_argument('--merger_dropout', type=float)
    parser.add_argument('--update_query_pos', type=str, help='provide true if using option and false if not')

    parser.add_argument('--sampler_steps', type=int, nargs='*')
    parser.add_argument('--sampler_lengths', type=int, nargs='*')
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--memory_bank_type', type=str)
    parser.add_argument('--query_denoise', type=float)
    
    # detection datasets for pseudo tracks training
    parser.add_argument('--append_det', type=str, help='whether to use detection data for training on pseudo tracks')
    parser.add_argument('--det_datasets', type=str, help='list of detection datasets to use for training on pseudo tracks')
    parser.add_argument('--det_dataset_splits', type=str, help='list of detection dataset splits to use for training on pseudo tracks')
    parser.add_argument('--data_root', type=str, help='')
    parser.add_argument('--mot_datasets', type=str, help='')
    parser.add_argument('--mot_dataset_splits', type=str, help='')
    # validation and test
    parser.add_argument('--inference_dataset', type=str, help='Name of dataset to run inference on')
    parser.add_argument('--inference_split', type=str, help='Name of dataset split to run inference on')
    parser.add_argument('--inference_model', type=str, help='path to model used for inference')
    parser.add_argument('--video_dir', type=str, help='path to video directory for video inference')
    parser.add_argument('--visualize_inference', type=str, help='provide true if using option and false if not')
    parser.add_argument('--evaluate_inference_mode', type=str, help='provide true if using option and false if not')
    parser.add_argument('--score_threshold_inference', type=float, help='')
    parser.add_argument('--newborn_threshold_inference', type=float, help='')
    parser.add_argument('--miss_tolerance', type=float, help='')
    # data
    return parser.parse_args()

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
                val_str = str(option_v).strip()
                if val_str.upper() == "TRUE":
                    config[option_k] = True
                elif val_str.upper() == "FALSE":
                    config[option_k] = False
                elif isinstance(option_v, str) and "," in option_v:
                    config[option_k] = [item.strip() for item in option_v.split(",") if item.strip()]
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