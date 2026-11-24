# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
# Based on: https://github.com/lhoyer/DAFormer and https://github.com/lhoyer/HRDA
# Modifications:
# - Adaptive Confidence Thresholding (ACT) driven by IMPACT weight maps
# - Source: SMOKE5K, Target: IMPACT-generated images, Val: Boreal Forest Fire
#
# All paths are relative to the ACT/ directory (run training from there).

_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer network architecture
    '../_base_/models/daformer_sepaspp_mitb5.py',
    # Data loading (overridden below)
    '../_base_/datasets/uda_gta_to_cityscapes_512x512.py',
    # Basic UDA self-training
    '../_base_/uda/dacs.py',
    # AdamW optimizer
    '../_base_/schedules/adamw.py',
    # Linear learning-rate warmup with subsequent linear decay
    '../_base_/schedules/poly10warm.py'
]

# Random seed
seed = 0
classes = ['Background', 'Smoke']
max_iters = 40000
val_interval = 4000

# UDA: DACS self-training + ACT
uda = dict(
    type='ACT',
    alpha=0.999,
    pseudo_threshold=0.968,   # base confidence threshold tau
    act_tau_min=0.5,          # threshold limit when the weight map agrees
    act_tau_max=1.0,          # threshold limit when the weight map disagrees
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    imnet_feature_dist_lambda=0,
    imnet_feature_dist_classes=[],
    imnet_feature_dist_scale_min_ratio=0.75,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2
)

# Dataset settings
dataset_type = 'CustomDataset'
data_root_source = 'dataset/source/'
data_root_target = 'dataset/target/images/'
weight_map_dir = 'dataset/target/weight_maps/'
data_root_val = 'dataset/val/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

src_scale = (1280, 960)
tgt_scale = (1280, 720)
crop_size = (640, 640)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='ResizeMask', img_scale=src_scale, keep_ratio=False),
    dict(type='RandomCropMask', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlipMask', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundleMask'),
    dict(type='CollectMask', keys=['img', 'gt_semantic_seg']),
]

target_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadWeightMap', data_root=weight_map_dir, suffix='.png'),
    dict(type='ResizeMask', img_scale=tgt_scale, keep_ratio=False),
    dict(type='RandomCropMask', crop_size=crop_size),
    dict(type='RandomFlipMask', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='PadWeightMap', pad_val=-1.0),
    dict(type='DefaultFormatBundleMask'),
    dict(type='CollectMask', keys=['img', 'weight_map']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADatasetACT',
        source=dict(
            type='CustomDataset',
            data_root=data_root_source,
            img_dir='images',
            ann_dir='masks',
            img_suffix='.jpg',
            seg_map_suffix='.png',
            classes=classes,
            palette=[[0, 0, 0], [255, 0, 0]],
            pipeline=train_pipeline),
        target=dict(
            type='CustomDataset',
            data_root=data_root_target,
            img_dir='',  # images directly under data_root_target
            img_suffix='.png',
            classes=classes,
            palette=[[0, 0, 0], [255, 0, 0]],
            pipeline=target_train_pipeline)
    ),
    val=dict(
        type='CustomDataset',
        data_root=data_root_val,
        img_dir='images',
        ann_dir='masks',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        classes=classes,
        palette=[[0, 0, 0], [255, 0, 0]],
        pipeline=test_pipeline),
    test=dict(
        type='CustomDataset',
        data_root=data_root_val,
        img_dir='images',
        ann_dir='masks',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        classes=classes,
        palette=[[0, 0, 0], [255, 0, 0]],
        pipeline=test_pipeline)
)

# HRDA model
model = dict(
    type='HRDAEncoderDecoder',
    pretrained=None,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/mit_b5.pth')
    ),
    decode_head=dict(
        type='HRDAHead',
        single_scale_head='DAFormerHead',
        attention_classwise=True,
        hr_loss_weight=0.1,
        num_classes=2),
    scales=[1, 0.5],
    hr_crop_size=[320, 320],
    feature_scale=0.5,
    crop_coord_divisible=8,
    hr_slide_inference=True,
    test_cfg=dict(
        mode='slide',
        batched_slide=True,
        stride=[320, 320],
        crop_size=[640, 640])
)

# Optimizer hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))

n_gpus = 1

runner = dict(type='IterBasedRunner', max_iters=max_iters)
checkpoint_config = dict(by_epoch=False, interval=val_interval, max_keep_ckpts=1)
evaluation = dict(interval=val_interval, metric='mIoU', distributed_eval=False,
                  efficient_test=False)

# Meta information for result analysis
name = 'act_hrda_smoke5k2boreal_mitb5'
exp = 'basic'
name_dataset = 'smoke5k2boreal'
name_architecture = 'hrda_daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'hrda_daformer_sepaspp'
name_uda = 'act'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_4k'
