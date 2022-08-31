## Mono3d
Recommended environments:


* GPU01 Docker container id：da8c4b19d3f6
* GPU02 Docker container id：5b49a1a545a0

```shell
python==3.8
cuda==11.1.1
mmcv==1.6.0
pytorch==1.10.0
```

# Create data
* Set up directory
```shell
cd mmdet3d
mkdir data
ln -s /home/yu.bai/Datasets/nuscenes/full nuscenesfull
ln -s /your/directory/to/save/datset/output nuscenesfullo
ln -s /home/yu.bai/Datasets/nuscenes/mini nuscenesmini
ln -s /your/another/directory/to/save/datset/output nuscenesminio
```
* Generate dataset's output file
```shell
sh scripts/create_data_single.sh
```

# Train and test
* Edit parameter in train_fusion.sh
    - CONFIG: model config path
    - EPOCH: which model to test
    - VISULIZATION_DIR: directory to save test results
```
CONFIG=configs/fusion/fusion_base_r18.py
EPOCH=work_dirs/fusion_base_r18/latest.pth
VISULIZATION_DIR=work_dirs/fusion_base_r18/results
```
* Run script
```shell
sh scripts/train_test_fusion.sh
```

# Todo: 
centerness_on_reg on head

Hard code to remove
resize_centers(self, results):
mask = [True]*6

backbone pretrain
val in data config
norm_on_bbox
not filtering cam bbox according to range
project_pts_to_img_depth online will increate data prepare time(from 0.02s to 0.2s), causing low gpu ultilization

point_cloud_range2 = [-50, -50, -5, 50, 50, 3]#hard code, to avoid duplicate point_cloud_range key

init_weights in fusion detector

scale_offset, scale_depth, scale_size = scale[0:3]

if num_gts == 0:
            return gt_labels.new_full((num_p

head.py
local yaw

get_targets
 if self.norm_on_bbox:

# loss
# need absolute due to possible negative delta x/y??
avg_factor

# change local yaw to global yaw for 3D nms
# mlvl_bboxes_3d = self.bbox_coder.decode_yaw(mlvl_bboxes_3d, mlvl_centers2d,
#                                          mlvl_dir_scores,
#                                          self.dir_offset)

return torch.concat([cx,cy,cz,w,l,h,ry], -1)??


centerness_on_reg

local yaw?
dists = torch.sqrt(torch.sum(bbox_targets_3d[..., :2]**2, dim=-1))


why cpu only??
if 'gt_bboxes_3d_cam_list' in results:
        key = 'gt_bboxes_3d_cam_list'
        results[key] = DC([res for res in results[key]], cpu_only=True)