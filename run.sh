#!/usr/bin/bash

#SBATCH -J grm-run
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=29G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g5
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

python main.py

exit 0