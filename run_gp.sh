#!/bin/bash

#SBATCH
#SBATCH --mail-type=END                      # Request status by email
#SBATCH --mail-user=yzs2@cornell.edu         # Email address to send results to.
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 1                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=20000M                         # server memory requested (per node; 1000M ~= 1G)
#SBATCH --gres=gpu:3090:1                    # Type/number of GPUs needed
#SBATCH --partition=kuleshov,gpu             # Request partition (gpu==medium priority; kuleshov==high priority)
#SBATCH --time=8:00:00                       # Set max runtime for job
#SBATCH --requeue                            # Requeue job


export PYTHONPATH="${PWD}:${PWD}/regression"

# shellcheck source=${HOME}/.bashrc
source "${CONDA_SHELL}"
conda activate tnp
cd ./regression || exit

python gp.py \
  --train_seed="${seed}" \
  --eval_seed="${seed}" \
  --model="${model}" \
  --expid="${expid}" \
  --mode="${mode}" \
  --lr="${lr}" \
  --min_lr="${min_lr}" \
  --num_steps="${num_steps}" \
  --annealer_mult="${annealer_mult}" \
  --max_num_ctx="${max_num_ctx}" \
  --min_num_ctx="${min_num_ctx}" \
  --max_num_tar="${max_num_tar}" \
  --min_num_tar="${min_num_tar}" \
  --eval_kernel="${eval_kernel}" \
  --eval_logfile="${eval_logfile}" \
  --resume "resume"

conda deactivate
