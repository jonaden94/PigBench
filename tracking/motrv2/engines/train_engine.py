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


"""
Train and eval functions used in main.py
"""
import os
import shutil
import math
import sys
from typing import Iterable
import torch
import time
from datasets.data_prefetcher import data_dict_to_cuda
import datasets.samplers as samplers
from torch.utils.data import DataLoader
from datasets import build_dataset
from models import build_model
from engines.eval_engine import evaluate_inference
from engines.inference_engine import inference
from util.tool import load_model
from util.misc import (MetricLogger, SmoothedValue,
                       reduce_dict, get_total_grad_norm, is_main_process, 
                       mot_collate_fn, match_name_keywords, save_on_master)
from util.sweeps import log_scores_sweep


def train(cfg):
    if cfg.frozen_weights is not None:
        assert cfg.masks, "Frozen training is meant for segmentation only"
    
    # model and dataset
    device = torch.device(cfg.device)
    model, criterion, postprocessors = build_model(cfg)
    model.to(device)
    model_without_ddp = model
    dataset_train = build_dataset(cfg=cfg)

    # dataloader
    if cfg.distributed:
        if cfg.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, cfg.batch_size, drop_last=True)
    collate_fn = mot_collate_fn
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=cfg.num_workers,
                                   pin_memory=True)
    
    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, cfg.lr_backbone_names) and not match_name_keywords(n, cfg.lr_linear_proj_names) and p.requires_grad],
            "lr": cfg.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, cfg.lr_backbone_names) and p.requires_grad],
            "lr": cfg.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, cfg.lr_linear_proj_names) and p.requires_grad],
            "lr": cfg.lr * cfg.lr_linear_proj_mult,
        }
    ]

    if cfg.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=cfg.lr, momentum=0.9,
                                    weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.lr,
                                      weight_decay=cfg.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.lr_drop)

    if cfg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu])
        model_without_ddp = model.module

    if cfg.frozen_weights is not None:
        checkpoint = torch.load(cfg.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    if cfg.pretrained is not None:
        model_without_ddp = load_model(model_without_ddp, cfg.pretrained)

    if cfg.resume:
        if cfg.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                cfg.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(cfg.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('########### Missing Keys: {}'.format(missing_keys), flush=True)
        if len(unexpected_keys) > 0:
            print('########### Unexpected Keys: {}'.format(unexpected_keys), flush=True)
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # This is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            cfg.override_resumed_lr_drop = True
            if cfg.override_resumed_lr_drop:
                print('########### Warning: (hack) cfg.override_resumed_lr_drop is set to True, so cfg.lr_drop would override lr_drop in resumed lr_scheduler.', flush=True)
                lr_scheduler.step_size = cfg.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            cfg.start_epoch = checkpoint['epoch'] + 1

    start_time = time.time()
    dataset_train.set_epoch(cfg.start_epoch)
    ############ START TRAINING
    best_hota = 0
    for epoch in range(cfg.start_epoch, cfg.epochs):
        if cfg.distributed:
            sampler_train.set_epoch(epoch)
        ############ TRAIN ONE EPOCH
        train_stats = train_one_epoch_mot(
            model, criterion, data_loader_train, optimizer, device, epoch, cfg.clip_max_norm, print_freq=cfg.print_freq)
        lr_scheduler.step()
        
        ############ SAVE CHECKPOINT
        checkpoint_path = os.path.join(cfg.outputs_dir, f'checkpoint{epoch}.pth')
        save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'cfg': cfg,
        }, checkpoint_path)
        if cfg.distributed:
            torch.distributed.barrier()
        dataset_train.step_epoch()
        
        ############ RUN INFERENCE WITH CURRENT CHECKPOINT
        if cfg.inference_dataset is not None:
            print(f'########### Epoch {epoch} eval: Running inference for validation', flush=True)
            start_time_eval = time.time()
            # dynamically set inference args based on current epoch
            cfg.outputs_dir_results = os.path.join(cfg.outputs_dir_results_base, f'epoch_{epoch}', 'results')
            cfg.inference_model = os.path.join(cfg.outputs_dir, f'checkpoint{epoch}.pth')
            inference(cfg)
            if cfg.distributed:
                torch.distributed.barrier()
                
            ############ EVALUATE INFERENCE RUN
            res_dict = evaluate_inference(cfg)
            end_time_eval = time.time()
            total_time_eval = end_time_eval - start_time_eval
            hours, remainder = divmod(int(total_time_eval), 3600)
            minutes, seconds = divmod(remainder, 60)
            formatted_time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
            print(f"########### Epoch {epoch} eval [{formatted_time_str}]: HOTA: {res_dict['HOTA']}, DetA: {res_dict['DetA']}, AssA: {res_dict['AssA']}, IDF1: {res_dict['IDF1']}", flush=True)
            
            # only save best model
            if is_main_process():
                if res_dict['HOTA'] > best_hota:
                    best_hota = res_dict['HOTA']
                    shutil.copy(cfg.inference_model, os.path.join(cfg.outputs_dir, "best.pth"))
                    os.remove(cfg.inference_model)
                else:
                    os.remove(cfg.inference_model)
            if cfg.distributed:
                torch.distributed.barrier()
                
    # log best hota once at the end of training
    if is_main_process():
        log_scores_sweep(best_hota=best_hota)
        
    # training finished
    total_time = time.time() - start_time
    hours, remainder = divmod(int(total_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    total_time_str = f"{hours:02}:{minutes:02}:{seconds:02}"
    print(f"########### Training finished in {total_time_str}", flush=True)



def train_one_epoch_mot(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, print_freq: int = 100):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for data_dict in metric_logger.log_every(data_loader, print_freq, header):
        data_dict = data_dict_to_cuda(data_dict, device)
        outputs = model(data_dict)

        loss_dict = criterion(outputs, data_dict)
        # print("iter {} after model".format(cnt-1))
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("########### Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        metric_logger.update(loss=loss_value)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        if is_main_process():
            log_scores_sweep(loss=loss_value, lr=optimizer.param_groups[0]["lr"], epoch=epoch, grad_norm=grad_total_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("########### Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
