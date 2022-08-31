#!/usr/bin/env bash

export PYTHONPATH=`pwd`:$PYTHONPATH
DATASET=nuscenes

python tools/create_data.py ${DATASET} \
            --root-path data/nuscenesfull \
            --out-dir data/nuscenesfullo \
            --extra-tag ${DATASET} \
            --max-sweeps 0 \
            --version v1.0
