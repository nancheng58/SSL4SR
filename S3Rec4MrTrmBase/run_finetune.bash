#!/bin/bash
#SBATCH -e result/Finetune_pop_05SP150_3layer.err
#SBATCH -o result/Finetune_pop_05SP150_3layer.out
#SBATCH -J 150_05_FT_SASRec

#SBATCH --partition=debug
#SBATCH --nodelist=gpu03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=999:00:00


conda activate torch1.1
python run_finetune_sample.py
