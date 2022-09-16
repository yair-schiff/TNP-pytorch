#!/bin/bash

#SBATCH
#SBATCH --mail-type=END                      # Request status by email
#SBATCH --mail-user=yzs2@cornell.edu         # Email address to send results to.
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 1                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=20000M                         # server memory requested (per node; 1000M ~= 1G)
#SBATCH --gres=gpu:1                         # Type/number of GPUs needed
#SBATCH --partition=gpu                 # Request partition (gpu==medium priority; kuleshov==high priority)
#SBATCH --time=8:00:00                       # Set max runtime for job


export PYTHONPATH="${PWD}:${PWD}/regression"

# shellcheck source=${HOME}/.bashrc
source "${CONDA_SHELL}"
conda activate tnp
cd ./regression || exit

python gp.py \
  --model="${model}" \
  --expid="${expid}" \
  --mode="${mode}" \
  --max_num_pts="${max_num_pts}" \
  --min_num_ctx="${min_num_ctx}" \
  --min_num_tar="${min_num_tar}" \
  --eval_kernel="${eval_kernel}" \
  --eval_logfile="${eval_logfile}" \
  --resume "resume"

conda deactivate
