#!/bin/bash

#SBATCH -e SGL_ml-pretrain.err

#SBATCH -o SGL_ml-pretrain.out
#SBATCH -J SGL_ml-pretrain

#SBATCH --partition=edu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2

#SBATCH --time=24:00:00

#SBATCH --mem=4G

python run_recbole.py --dataset='ML-1M' --model='SGL'

