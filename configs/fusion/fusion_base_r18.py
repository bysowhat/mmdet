_base_ = [
    '../_base_/datasets/nus-mm.py', '../_base_/models/fusion.py',
    '../_base_/schedules/mmdet_schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=6)
# optimizer
optimizer = dict(
    lr=0.003, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 10,
    step=[10, 20])
total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
evaluation = dict(interval=999)
checkpoint_config = dict(interval=2, max_keep_ckpts=15)
log_config = dict(
    interval=50
    )
