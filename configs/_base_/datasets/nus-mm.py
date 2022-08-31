point_cloud_range = [-50, -50, -5, 50, 50, 3]
dataset_type = 'NuScenesMMDataset'
data_root = 'data/nuscenesminio/'
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
num_views=6

img_scale = (800, 448)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4),
    dict(
        type='LoadMultiViewImageFromFilesMono',
        project_pts_to_img_depth=False),
    dict(
        type='LoadAnnotations3DMM',
        with_bbox=True,
        with_label=True,
        with_attr_label=False,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilterMM', point_cloud_range=point_cloud_range),
    dict(type='MyResize', img_scale=img_scale, keep_ratio=True),
    dict(type='MyNormalize', **img_norm_cfg),
    dict(type='MyPad', size_divisor=32),
    dict(type='DefaultFormatBundle3DMM', class_names=class_names),
    dict(
        type='Collect3DMM',
        keys=[
            'img', 'points', 'coords_list',
            'gt_bboxes2d_cam_list', 'gt_labels2d_cam_list', 
            'gt_bboxes_3d_cam_list', 'centers2d_cam_list', 'depths_cam_list', 'gt_labels_3d_cam_list',
            'gt_bboxes_3d_bev', 'gt_labels_3d_bev', 'camera_mask'
        ]),
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4),
    dict(
        type='LoadMultiViewImageFromFilesMono',
        project_pts_to_img_depth=False),
    dict(type='MyResize', img_scale=img_scale, keep_ratio=True),
    dict(type='MyNormalize', **img_norm_cfg),
    dict(type='MyPad', size_divisor=32),
    dict(type='DefaultFormatBundle3DMM', class_names=class_names),
    dict(
        type='Collect3DMM',
        keys=[
            'img', 'points', 'coords_list'
        ]),
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['img'])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        num_views=num_views,
        ann_file=data_root + 'nuscenes_infos_train_mono3d.pkl',
        img_prefix=None,
        classes=class_names,
        pipeline=train_pipeline,
        modality=input_modality,
        filter_empty_gt=True,
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        num_views=num_views,
        ann_file=data_root + 'nuscenes_infos_val_mono3d.pkl',
        img_prefix=None,
        classes=class_names,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=False),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        num_views=num_views,
        ann_file=data_root + 'nuscenes_infos_val_mono3d.pkl',
        img_prefix=None,
        classes=class_names,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=False))
evaluation = dict(interval=999)
