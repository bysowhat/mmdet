#!/usr/bin/env bash

export PYTHONPATH=`pwd`:$PYTHONPATH
CONFIG_FILE=configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py
CHECKPOINT_FILE=checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_20210715_235813-4bed5239.pth
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --show --show-dir /home/yu.bai/Projects/GitHub/mmdetection3d/outs