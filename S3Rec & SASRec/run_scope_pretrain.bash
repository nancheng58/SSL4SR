#!/bin/bash
#SBATCH -e result/Yelp.err
#SBATCH -o result/Yelp.out
#SBATCH -J S3Rec_pretrain

#SBATCH --partition=debug 
#SBATCH --nodelist=gpu03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=999:00:00


conda activate torch1.1
python run_pretrain.py