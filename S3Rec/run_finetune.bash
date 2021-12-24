#!/bin/bash
#SBATCH -e result/Finetune-Beauty-sample_150.err
#SBATCH -o result/Finetune-Beauty-sample_150.out
#SBATCH -J S3Rec_ft_sample

#SBATCH --partition=debug 
#SBATCH --nodelist=gpu03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=999:00:00


conda activate torch1.1
python run_finetune_sample.py