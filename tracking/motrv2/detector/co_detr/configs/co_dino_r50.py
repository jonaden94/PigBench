# Modified from the MMDetection project (https://github.com/open-mmlab/mmdetection/tree/main/projects/CO-DETR/configs/codino)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

_base_ = './co_dino_r50_lsj.py'

model = dict(
    use_lsj=False, data_preprocessor=dict(pad_mask=False, batch_augments=None))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    dataset=dict(
        metainfo=_base_.metainfo,
        _delete_=True,
        type=_base_.dataset_type,
        data_root=_base_.data_root,
        ann_file=_base_.ann_file_train,
        data_prefix=dict(img=_base_.data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        backend_args=_base_.backend_args))

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(dataset=dict(metainfo=_base_.metainfo, pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(metainfo=_base_.metainfo, pipeline=test_pipeline))
