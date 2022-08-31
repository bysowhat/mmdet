#!/usr/bin/env bash

export PYTHONPATH=`pwd`:$PYTHONPATH
CONFIG_FILE=configs/fusion/fusion_base_r18.py
CHECKPOINT_FILE=work_dirs/fusion_base_r18/epoch_12.pth
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval bo