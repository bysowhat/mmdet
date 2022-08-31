point_cloud_range2 = [-50, -50, -5, 50, 50, 3]#hard code, to avoid duplicate point_cloud_range key
voxel_size = [0.25, 0.25, 1]
height_dim = 2
in_channels0 = 2
bev_channel = int((point_cloud_range2[height_dim+3] - point_cloud_range2[height_dim])/voxel_size[height_dim])
model = dict(
    type='Fusion',
    pts_voxel_layer=dict(
        max_num_points=8,
        point_cloud_range=point_cloud_range2,
        voxel_size=voxel_size,
        max_voxels=(30000, 40000)),
    pts_voxel_encoder=dict(
        type='AVODBEV',
        height_dim=height_dim,
        reflect_dim=3,
        in_channels=4,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range2,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)),
    pts_middle_encoder=dict(
        type='AVODScatter', 
        in_channels0=in_channels0, 
        in_channels1=1, 
        point_cloud_range=point_cloud_range2,
        voxel_size=voxel_size,
        height_dim=height_dim,
        output_shape=[bev_channel, 400, 400]),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/home/a/Projects_data/mmdet/pretrain/resnet50-19c8e357.pth')),
    img_neck=dict(
        type='FPN',
        # in_channels=[64, 128, 256, 512],
        in_channels=[256, 512, 1024, 2048],
        out_channels=512,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    img_bbox_head=dict(
        type='FCOSMono3DHead',
        num_classes=10,
        in_channels=512,
        stacked_convs=2,
        feat_channels=128,
        use_direction_classifier=True,
        diff_rad_by_sin=True,
        bbox_code_size=7,
        pred_attrs=False,
        pred_velo=False,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        strides=[8, 16, 32, 64, 128],
        group_reg_dims=(2, 1, 3, 1),  # offset, depth, size, rot, velo
        cls_branch=(128, ),
        reg_branch=(
            (128, ),  # offset
            (128, ),  # depth
            (128, ),  # size
            (128, ),  # rot
        ),
        dir_branch=(128, ),
        attr_branch=(128, ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_attr=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        bbox_coder=dict(type='FCOS3DBBoxCoder', code_size=9),
        norm_on_bbox=True,
        centerness_on_reg=True,
        center_sampling=True,
        conv_bias=True,
        dcn_on_last_conv=False),
    pts_bbox_head=dict(
        type='FCOSMono3DHeadPts',
        num_classes=10,
        in_channels=512,
        stacked_convs=2,
        feat_channels=512,
        use_direction_classifier=True,
        diff_rad_by_sin=True,
        bbox_code_size=4,
        pred_attrs=False,
        pred_velo=False,
        point_cloud_range=point_cloud_range2,
        voxel_size=voxel_size,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        strides=[8, 16, 32, 64, 128],
        group_reg_dims=(4, 1, 1, 1),  # (left, top, right, bottom), gt_z, gt_h, gt_ry
        cls_branch=(128, ),
        reg_branch=(
            (128, ),  # (left, top, right, bottom)
            (128, ),  # gt_z
            (128, ),  # gt_h
            (128, ),  # gt_ry
        ),
        dir_branch=(128, ),
        attr_branch=(128, ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_bbox_2d=dict(type='IoULoss', loss_weight=0.6),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_attr=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        bbox_coder2d=dict(type='DistancePointBBoxCoder'),
        bbox_coder3d=dict(type='FCOS3DBBoxCoderPts', code_size=4),
        norm_on_bbox=True,
        centerness_on_reg=False,
        center_sampling=True,
        conv_bias=True, 
        dcn_on_last_conv=False),
    train_cfg=dict(
        allowed_border=0,
        code_weight_img=[1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0],
        code_weight_pts=[0.1, 0.1, 0.1, 0.1, 0.2, 1.0, 1.0],
        branch_weight=[1.0, 2.0, 1.0],#img branch loss weight, pts branch loss weight, fusion branch loss weight
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=1000,
        nms_thr=0.8,
        score_thr=0.05,
        min_bbox_size=0,
        max_per_img=200))
