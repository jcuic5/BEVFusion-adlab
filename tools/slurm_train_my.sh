#!/usr/bin/env bash

set -x

CONFIG=$1
WORK_DIR=$2
PY_ARGS=${@:3}
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun --partition=general \
    --nodelist=ewi3 \
    --qos=long \
    --job-name=centerpoint \
    --cpus-per-task=64 \
    --mem=300000 \
    --time=120:00:00 \
    --gpus-per-node=2 \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" ${PY_ARGS}

#./tools/slurm_train_my.sh configs/*** work_dirs/***
# sinteractive --nodelist=influ1 --gpus-per-node=2