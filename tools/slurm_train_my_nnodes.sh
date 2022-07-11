#!/usr/bin/env bash

set -x

CONFIG=$1
WORK_DIR=$2
PY_ARGS=${@:3}
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
NCCL_DEBUG=INFO \
NCCL_SOCKET_IFNAME="campus" \
srun --partition=general \
    --job-name=centerpoint \
    --nodelist=gpu05,gpu06 \
    --nodes=2 \
    --gpus-per-node=a40:2 \
    --qos=long \
    --time=120:00:00 \
    --mem=350000 \
    --cpus-per-task=60 \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" ${PY_ARGS}

#GPUS=8 GPUS_PER_NODE=2 CPUS_PER_TASK=32 ./tools/slurm_train_my_nnodes.sh general centerpoint configs/_my_configs_/centerpoint_02pillar_second_secfpn_8x2_cyclic_20e_nus work_dirs/Wed-Jul06-2242

# NCCL_SOCKET_IFNAME="131.180.180.7" \