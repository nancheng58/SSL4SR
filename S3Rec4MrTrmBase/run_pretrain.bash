#!/bin/bash
#SBATCH -e result/ml-1m_0.5SP_drop0.2.err
#SBATCH -o result/ml-1m_0.5SP_dorp0.2.out
#SBATCH -J 0.5SP_S3Rec

#SBATCH --partition=debug 
#SBATCH --nodelist=gpu05
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=999:00:00


conda activate torch1.1
python run_pretrain.py