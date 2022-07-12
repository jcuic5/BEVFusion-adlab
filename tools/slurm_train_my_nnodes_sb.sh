#!/usr/bin/env bash
#SBATCH --partition=general
#SBATCH --job-name=bevf_centerpoint
#SBATCH --qos=long
#SBATCH --time=120:00:00
#SBATCH --nodelist=gpu02,gpu03,gpu04,gpu05
#SBATCH --nodes=4
#SBATCH --cpus-per-task=60
#SBATCH --gpus-per-node=a40:2
#SBATCH --mem=350000
##SBATCH --constraint=gpumem32

module use /opt/insy/modulefiles
module load miniconda
module load cuda/11.6
conda activate bevfusion

CONFIG=$1
WORK_DIR=$2
PY_ARGS=${@:3}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
NCCL_DEBUG=INFO \
NCCL_SOCKET_IFNAME="campus" \
python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" ${PY_ARGS}
