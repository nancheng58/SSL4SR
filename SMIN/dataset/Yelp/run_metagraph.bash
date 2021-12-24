#!/bin/bash
#SBATCH -e result/Yelp.err
#SBATCH -o result/Yelp.out
#SBATCH -J SMIN_dataGen

#SBATCH --partition=debug 
#SBATCH --nodelist=gpu03
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=4
#SBATCH --time=999:00:00


conda activate torch1.5
python GenerateMetaPath.py