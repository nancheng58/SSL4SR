#!/bin/bash
#SBATCH -e result/DVD.err
#SBATCH -o result/DVD.out
#SBATCH -J SMIN_train

#SBATCH --partition=debug 
#SBATCH --nodelist=gpu03
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --time=999:00:00


conda activate torch1.5
python main.py --dataset CiaoDVD --hide_dim 16 --layer_dim [16] --lr 0.05 --reg 0.05  --lambda1 0.06 --lambda2 0.002 