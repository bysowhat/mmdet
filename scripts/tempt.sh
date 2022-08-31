#!/usr/bin/env bash

export PYTHONPATH=`pwd`:$PYTHONPATH
DATASET=nuscenes

python tools/create_data.py ${DATASET} \
            --root-path data/nuscenesfull \
            --out-dir data/nuscenesfullo \
            --extra-tag ${DATASET} \
            --max-sweeps 0 \
            --version v1.0

CONFIG=configs/fusion/fusion_base_r18.py
EPOCH=work_dirs/fusion_base_r18/latest.pth
VISULIZATION_DIR=work_dirs/fusion_base_r18/results
GPUS=8
MASTER_ADDRESS=127.0.0.6
PORT=10014
GPU_LIST='0,1,2,3,4,5,6,7'

CUDA_VISIBLE_DEVICES=$GPU_LIST python3 -m torch.distributed.launch --master_addr=$MASTER_ADDRESS --nproc_per_node=$GPUS --master_port=$PORT \
    ./tools/train.py $CONFIG --launcher pytorch
python3 ./tools/test.py $CONFIG $EPOCH --eval box --show --show-dir $VISULIZATION_DIR --show-score-thr 0.1