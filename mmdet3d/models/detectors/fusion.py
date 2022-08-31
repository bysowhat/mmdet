# Copyright (c) OpenMMLab. All rights reserved.
from operator import inv
import os
import mmcv
import numpy as np
import torch

from torch.nn import functional as F
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from ..builder import DETECTORS, build_backbone, build_head, build_neck, build_voxel_encoder, build_middle_encoder
from .base import BaseDetector
from mmdet3d.core import (Box3DMode, Coord3DMode, CameraInstance3DBoxes, bbox3d2result,
                          show_multi_modality_result, show_result)
from mmdet3d.ops import Voxelization
from mmcv.cnn import ConvModule


@DETECTORS.register_module()
class Fusion(BaseDetector):
    r""" Early fusion for 3D object detection.
    """

    def __init__(self,
                 pts_voxel_layer,
                 pts_voxel_encoder,
                 pts_middle_encoder,
                 img_backbone,
                 img_neck,
                 img_bbox_head,
                 pts_bbox_head,
                 fuse_bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Fusion, self).__init__(None)
        self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        img_bbox_head.update(train_cfg=train_cfg)
        img_bbox_head.update(test_cfg=test_cfg)
        pts_bbox_head.update(train_cfg=train_cfg)
        pts_bbox_head.update(test_cfg=test_cfg)
        fuse_bbox_head.update(train_cfg=train_cfg)
        fuse_bbox_head.update(test_cfg=test_cfg)
        self.img_bbox_head = build_head(img_bbox_head)
        self.pts_bbox_head = build_head(pts_bbox_head)
        self.fuse_bbox_head = build_head(fuse_bbox_head)
        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        if pts_voxel_encoder:
            self.pts_voxel_encoder = build_voxel_encoder(pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = build_middle_encoder(pts_middle_encoder)
        self.pts_stem = self.reduc_conv = ConvModule(
                self.pts_middle_encoder.output_shape[0]+self.pts_middle_encoder.in_channels0,
                3,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
    
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        input_shape = img.shape[-2:]
        # update real input shape of each single img
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        if img.dim() == 5 and img.size(0) == 1:
            img.squeeze_(0)
        elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            img = img.view(B * N, C, H, W)
        x = self.img_backbone(img.float())
        x = self.img_neck(x)
        return x

    def extract_pts_feat(self, pts, img_feats, img_metas, gt_bboxes_3d=None):
        """Extract features of points."""
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                img_feats, img_metas)

        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(*voxel_features, coors, batch_size)
        bev = x.clone()
        x = self.pts_stem(x.float())
        x = self.img_backbone(x)
        x = self.img_neck(x)
        return x, bev

    def reformat(self, *args):
        new_results = []
        for item in args[1:]:
            if item is None:
                new_results.append(item)
            else:
                #Batch
                new_list = []
                for item2 in item:
                    #Views
                    for item3 in item2:
                        new_list.append(item3)
                new_results.append(new_list)
        return new_results

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train_img(self, 
                          gt_bboxes2d_cam_list, 
                          gt_labels2d_cam_list,
                          gt_bboxes_3d_cam_list, gt_labels_3d_cam_list,
                          centers2d_cam_list,
                          depths_cam_list,
                          attr_labels, 
                          gt_bboxes_ignore, 
                          camera_mask,
                          img_feats,
                          img_metas):
        gt_bboxes2d_cam_list, gt_labels2d_cam_list, \
        gt_bboxes_3d_cam_list, gt_labels_3d_cam_list, \
        centers2d_cam_list, depths_cam_list, \
        attr_labels, gt_bboxes_ignore, camera_mask = self.reformat(self, gt_bboxes2d_cam_list,
                                                  gt_labels2d_cam_list, gt_bboxes_3d_cam_list,
                                                  gt_labels_3d_cam_list, centers2d_cam_list, depths_cam_list,
                                                  attr_labels, gt_bboxes_ignore, camera_mask)
        loss = self.img_bbox_head.forward_train(img_feats, img_metas, gt_bboxes2d_cam_list,
                                                  gt_labels2d_cam_list, gt_bboxes_3d_cam_list,
                                                  gt_labels_3d_cam_list, centers2d_cam_list, depths_cam_list,
                                                  attr_labels, camera_mask, gt_bboxes_ignore)
        return loss

    def forward_train_pts(self,
                          bev,
                          gt_bboxes_3d_bev, 
                          gt_labels_3d_bev,
                          gt_bboxes_ignore, 
                          pts_feats,
                          img_metas):
        loss = self.pts_bbox_head.forward_train(bev,
                                                pts_feats,
                                                img_metas,
                                                gt_bboxes_3d_bev,
                                                gt_labels_3d_bev,
                                                gt_bboxes_ignore)
        return loss
    
    def forward_train_fuse(self,
                          bev,
                          gt_bboxes_3d_bev, 
                          gt_labels_3d_bev,
                          gt_bboxes_ignore, 
                          feats,
                          img_metas):
        loss = self.fuse_bbox_head.forward_train(bev,
                                                feats,
                                                img_metas,
                                                gt_bboxes_3d_bev,
                                                gt_labels_3d_bev,
                                                gt_bboxes_ignore)
        return loss
    
    def parase_loss(self, losses):
        weights = self.train_cfg['branch_weight']
        final_loss = {}
        prefix = ['img_', 'pts_', 'fus_']
        for i in range(3):
            if losses[i] is None:
                    continue
            for key, val in losses[i].items():
                assert prefix[i]+key not in final_loss
                final_loss[prefix[i]+key] = val * weights[i]
        return final_loss

    def unique_coord(self, coord):
        
        def _unique(x, dim=None):
            unique, inverse = torch.unique(x, 
                                           sorted=True,
                                           return_inverse=True,
                                           dim=dim)
            perm = torch.arange(inverse.size(0), 
                                dtype=inverse.dtype,
                                device=inverse.device)
            inverse, perm = inverse.flip([0]), perm.flip([0])
            return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)

        def _unique_coord(img_coord, step=1000):
            tempt = img_coord[:, 0]*step + img_coord[:, 1]
            _, unique_index = _unique(tempt)
            return unique_index

        img_coord = coord[:, :2]
        pts_coord = coord[:, 2:]
        img_index = _unique_coord(img_coord)
        pts_index = _unique_coord(pts_coord)

        return coord[pts_index, :]

    def caculate_pos(self, coords, i_lvl):
        img_stride = self.img_bbox_head.strides[i_lvl]
        pts_stride = self.pts_bbox_head.strides[i_lvl]
        voxel_size = self.pts_bbox_head.voxel_size
        point_cloud_range = self.pts_bbox_head.point_cloud_range
        point_cloud_range = torch.tensor(np.asarray(point_cloud_range[:2]), device=coords[0].device)
        voxel_size = torch.tensor(np.asarray(voxel_size[:2]), device=coords[0].device)
        img_nums = len(coords)
        
        unique_coord_list = []
        for i_img in range(img_nums):
            coord = coords[i_img]
            coord[:, 0] /= img_stride
            coord[:, 1] /= img_stride
            coord[:, 2:4] = (coord[:, 2:4] - point_cloud_range[None])/(voxel_size*pts_stride)
            coord = torch.round(coord).long()
            unique_coord = self.unique_coord(coord)
            unique_coord_list.append(unique_coord)
        return unique_coord_list

    def fuse_feats_coord(self, img_feats, pts_feat, coord_pos_list, num_imgs):
        fuse_feat = pts_feat.new_zeros(pts_feat.size())
        fuse_feat_mask = pts_feat.new_ones(pts_feat.shape[1], pts_feat.shape[2])
        
        pts_h = pts_feat.shape[1]
        pts_w = pts_feat.shape[2]
        img_h = img_feats[0].shape[1]
        img_w = img_feats[0].shape[2]

        for i in range(num_imgs):
            img_feat = img_feats[i]
            coord_pos = coord_pos_list[i]
            mask = pts_feat.new_ones(coord_pos.shape[0], dtype=torch.bool)
            mask = torch.logical_and(mask, coord_pos[:,0] < img_w)
            mask = torch.logical_and(mask, coord_pos[:,1] < img_h)
            mask = torch.logical_and(mask, coord_pos[:,2] < pts_w)
            mask = torch.logical_and(mask, coord_pos[:,3] < pts_h)
            coord_pos = coord_pos[mask, :]

            img_feat = img_feat.view(-1, img_h*img_w)
            img_index = coord_pos[:,1] * img_w + coord_pos[:,0]
            img_feat_gather = img_feat[:, img_index]
            
            pts_feat_tempt = pts_feat.new_zeros(pts_feat.size())
            pts_feat_mask = pts_feat.new_zeros(pts_feat.shape[1]*pts_feat.shape[2])
           
            pts_feat_tempt = pts_feat_tempt.view(-1, pts_h*pts_w)
            pts_index = coord_pos[:,3] * pts_w + coord_pos[:,2]
            pts_feat_tempt[:, pts_index] = img_feat_gather
            pts_feat_mask[pts_index] = 1.0
            fuse_feat += pts_feat_tempt.view(-1, pts_h, pts_w)
            fuse_feat_mask += pts_feat_mask.view(pts_h, pts_w)
        return fuse_feat / fuse_feat_mask

    def fuse_feature(self, img_feats, pts_feats, coords_list):
        num_lvl = len(img_feats)
        bs = pts_feats[0].shape[0]
        num_imgs = int(img_feats[0].shape[0]/bs)

        fuse_feats = []
        for i_lvl in range(num_lvl):
            img_fuse_feats = []
            for i_b in range(bs):
                coords = coords_list[i_b]
                # Todo: Adjust positon selection method
                coord_pos = self.caculate_pos(coords, i_lvl)
                img_fuse_feats.append(self.fuse_feats_coord(img_feats[i_lvl][num_imgs*i_b:num_imgs*(i_b+1)], pts_feats[i_lvl][i_b], coord_pos, num_imgs))
            fuse_feats.append(torch.stack(img_fuse_feats)+pts_feats[i_lvl])
        return fuse_feats

    def forward_train(self,
                      img,
                      img_metas,
                      points,
                      coords_list,
                      gt_bboxes2d_cam_list,
                      gt_labels2d_cam_list,
                      gt_bboxes_3d_cam_list,
                      centers2d_cam_list,
                      depths_cam_list,
                      gt_labels_3d_cam_list,
                      gt_bboxes_3d_bev,
                      gt_labels_3d_bev,
                      camera_mask,
                      attr_labels=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_3d (list[Tensor]): Each item are the 3D truth boxes for
                each image in [x, y, z, x_size, y_size, z_size, yaw, vx, vy]
                format.
            gt_labels_3d (list[Tensor]): 3D class indices corresponding to
                each box.
            centers2d (list[Tensor]): Projected 3D centers onto 2D images.
            depths (list[Tensor]): Depth of projected centers on 2D images.
            attr_labels (list[Tensor], optional): Attribute indices
                corresponding to each box
            gt_bboxes_ignore (list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        ###################
        # img branch
        ###################
        img_feats = self.extract_img_feat(img, img_metas)
        img_losses = self.forward_train_img(gt_bboxes2d_cam_list, 
                                            gt_labels2d_cam_list,
                                            gt_bboxes_3d_cam_list, 
                                            gt_labels_3d_cam_list,
                                            centers2d_cam_list,
                                            depths_cam_list,
                                            attr_labels, 
                                            gt_bboxes_ignore, 
                                            camera_mask,
                                            img_feats,
                                            img_metas)
        ###################
        # pts branch
        ###################
        pts_feats, bevimg = self.extract_pts_feat(points, img_feats, img_metas)
        pts_losses = self.forward_train_pts(bevimg,
                                            gt_bboxes_3d_bev, 
                                            gt_labels_3d_bev,
                                            gt_bboxes_ignore, 
                                            pts_feats,
                                            img_metas)

        ###################
        # fuse branch
        ###################
        fuse_feats = self.fuse_feature(img_feats, pts_feats, coords_list)
        fuse_losses = self.forward_train_fuse(bevimg,
                                              gt_bboxes_3d_bev, 
                                              gt_labels_3d_bev,
                                              gt_bboxes_ignore, 
                                              fuse_feats,
                                              img_metas)

        final_loss = self.parase_loss([img_losses, pts_losses, fuse_losses])
        return final_loss


    def forward_test(self, imgs, img_metas, points, coords_list, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        imgs = [imgs]
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        # for img, img_meta in zip(imgs, img_metas):
        #     batch_size = img.shape[0]
        #     for img_id in range(batch_size):
        #         img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            return self.simple_test(imgs[0], img_metas, points, coords_list, **kwargs)
        else:
            raise NotImplementedError()
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    def extract_feat(self, img):
        """Have to implement extract_feat in father class.
        Todo: Edit father class.
        """
        pass

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_img_feat(img)
        outs = self.bbox_head(x)
        return outs

    def simple_test(self, imgs, img_metas, points, coords_list, rescale=False, **kwargs):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas dict: image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        # Img only predict
        img_feat = self.extract_img_feat(imgs, img_metas)
        img_results_list = self.img_bbox_head.simple_test(
            img_feat, img_metas, box_type_3d='box_type_3d_cam', rescale=rescale)
        img_bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels, det_attrs)
            for det_bboxes, det_scores, det_labels, det_attrs in img_results_list
        ]

        # Lidar only predict
        pts_feats, bevimg = self.extract_pts_feat(points, img_feat, img_metas)
        pts_results_list = self.pts_bbox_head.simple_test(bevimg, pts_feats, img_metas, box_type_3d='box_type_3d_lidar')
        pts_bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels, det_attrs)
            for det_bboxes, det_scores, det_labels, det_attrs in pts_results_list
        ]

        # Fusion predict
        fuse_feats = self.fuse_feature(img_feat, pts_feats, coords_list)
        fuse_results_list = self.pts_bbox_head.simple_test(bevimg, fuse_feats, img_metas, box_type_3d='box_type_3d_lidar')
        fuse_bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels, det_attrs)
            for det_bboxes, det_scores, det_labels, det_attrs in fuse_results_list
        ]
        return img_bbox_results, pts_bbox_results, fuse_bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            *outs, img_metas, with_nms=with_nms)

        return det_bboxes, det_labels
    
    def show_img_results(self,
                    data, 
                    result,
                    out_dir,
                    show,
                    score_thr):
        img_num = len(data['img_metas']._data[0][0]['filename'])
        assert img_num == len(result)
        for img_id in range(img_num):
            if isinstance(data['img_metas'], DC):
                img_filename = data['img_metas']._data[0][0]['filename'][img_id]
                cam2img = data['img_metas']._data[0][0]['cam2img_ori'][img_id]
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'])} "
                    f'for visualization!')
            img = mmcv.imread(img_filename)
            file_name = os.path.split(img_filename)[-1].split('.')[0]

            assert out_dir is not None, 'Expect out_dir, got none.'

            pred_bboxes = result[img_id]['boxes_3d']
            assert isinstance(pred_bboxes, CameraInstance3DBoxes), \
                f'unsupported predicted bbox type {type(pred_bboxes)}'

            pred_scores = result[img_id]['scores_3d']
            mask = pred_scores > score_thr
            pred_bboxes = pred_bboxes[mask]

            show_multi_modality_result(
                img,
                None,
                pred_bboxes,
                cam2img,
                out_dir,
                file_name,
                'camera',
                show=show)

    def show_pts_results(self,
                    data, 
                    result,
                    out_dir,
                    show,
                    score_thr,
                    snapshot):
        for batch_id in range(len(result)):
            if isinstance(data['points'], DC):
                points = data['points']._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data['points'][0], torch.Tensor):
                raise NotImplementedError()
                points = data['points'][0][batch_id]
            else:
                ValueError(f"Unsupported data type {type(data['points'][0])} "
                           f'for visualization!')
            if isinstance(data['img_metas'], DC):
                pts_filename = data['img_metas']._data[0][batch_id][
                    'pts_filename']
                box_mode_3d = data['img_metas']._data[0][batch_id][
                    'box_mode_3d_lidar']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                raise NotImplementedError()
                pts_filename = data['img_metas'][0][batch_id]['pts_filename']
                box_mode_3d = data['img_metas'][0][batch_id]['box_mode_3d']
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} "
                    f'for visualization!')
            file_name = os.path.split(pts_filename)[-1].split('.')[0]

            assert out_dir is not None, 'Expect out_dir, got none.'
            inds = result[batch_id]['scores_3d'] > score_thr
            pred_bboxes = result[batch_id]['boxes_3d'][inds]

            # for now we convert points and bbox into depth mode
            if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d
                                                  == Box3DMode.LIDAR):
                points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                                   Coord3DMode.DEPTH)
                pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d,
                                                Box3DMode.DEPTH)
            elif box_mode_3d != Box3DMode.DEPTH:
                ValueError(
                    f'Unsupported box_mode_3d {box_mode_3d} for conversion!')

            pred_bboxes = pred_bboxes.tensor.cpu().numpy()
            show_result(points, None, pred_bboxes, out_dir, file_name, show=show, snapshot=snapshot)

    def show_results(self,
                    data, 
                    img_result,
                    pts_result,
                    fuse_result,
                    out_dir,
                    show,
                    score_thrs):
        self.show_img_results(data, 
                              img_result,
                              out_dir+'_img',
                              show,
                              score_thrs[0])
        
        self.show_pts_results(data, 
                              pts_result,
                              out_dir+'_pts',
                              show,
                              score_thrs[1],
                              snapshot=True)
        
        self.show_pts_results(data, 
                              fuse_result,
                              out_dir+'_fus',
                              show,
                              score_thrs[2],
                              snapshot=True)
           
           