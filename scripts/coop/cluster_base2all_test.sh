#!/bin/bash

# custom config
DATA=/hdd/hdd2/sch/DATA
TRAINER=CoOp

DATASET=$1

# CFG=vit_b16_c4_ep10_batch1_ctxv1
CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
SHOTS=$2
SEED=$3
LOADEP=200
SUB=all


COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=output/cluster_base2new/train_base/${COMMON_DIR}
DIR=output/cluster_base2new/test_${SUB}/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi