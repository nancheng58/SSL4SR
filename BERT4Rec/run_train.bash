#!/bin/bash
#SBATCH -e result/bert_B_train_augment_maskp=0.1.err
#SBATCH -o result/bert_B_train_augment_maskp=0.1.out
#SBATCH -J bert_B

#SBATCH --partition=debug
#SBATCH --nodelist=gpu06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=999:00:00


conda activate torch1.8
python main.py --bert_mask_prob=0.1
