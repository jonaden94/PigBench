# Modified from the MMYolo project (https://github.com/open-mmlab/mmyolo)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

_base_ = './yolox_s.py'

load_from = 'data/pretrained_weights/yolox_coco/yolox_s_fast_8xb32-300e-rtmdet-hyp_coco_20230210_134645-3a8dfbd7.pth'
# ========================modified parameters======================
train_batch_size_per_gpu = 6
batch_augments_interval = 1
num_last_epochs = _base_.num_last_epochs
base_lr = 0.004
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
ema_momentum = 0.0002

# ============================== Unmodified in most cases ===================
model = dict(
    data_preprocessor=dict(batch_augments=[
        dict(
            type='YOLOXBatchSyncRandomResize',
            random_size_range=_base_.random_size_range,
            size_divisor=32,
            interval=batch_augments_interval)
    ]))

param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 5 to 285 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=5,
        T_max=_base_.max_epochs - num_last_epochs,
        end=_base_.max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last num_last_epochs epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=_base_.max_epochs - num_last_epochs,
        end=_base_.max_epochs,
    )
]

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        new_train_pipeline=_base_.train_pipeline_stage2,
        priority=48),
    dict(type='mmdet.SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=ema_momentum,
        update_buffers=True,
        strict_load=False,
        priority=49)
]

train_dataloader = dict(batch_size=train_batch_size_per_gpu)
train_cfg = dict(dynamic_intervals=[(_base_.max_epochs - num_last_epochs, 1)])
auto_scale_lr = dict(base_batch_size=256, enable=True)
