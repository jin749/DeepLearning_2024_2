#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0
# bash scripts/coop/cluster_main.sh dtd -1 1

# custom config
DATA=/hdd/hdd2/sch/DATA
TRAINER=CoOp

DATASET=$1
CFG=vit_b16  # config file
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=$2  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)

SEED=$3

DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}

python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
TRAINER.COOP.N_CTX ${NCTX} \
TRAINER.COOP.CSC ${CSC} \
TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
DATASET.NUM_SHOTS ${SHOTS}
