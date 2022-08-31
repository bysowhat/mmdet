#!/usr/bin/env bash

CONFIG=configs/fusion/fusion_base_r18.py
EPOCH=work_dirs/fusion_base_r18/latest.pth
VISULIZATION_DIR=work_dirs/results
export PYTHONPATH='pwd':$PYTHONPATH

python3 ./tools/train.py $CONFIG
python3 ./tools/test.py $CONFIG $EPOCH --eval box --show --show-dir $VISULIZATION_DIR --show-score-thr 0.1
