#!/bin/bash
#SBATCH -e result16/Tmall.err
#SBATCH -o result16/Tmall.out
#SBATCH -J DHCN_torch16

#SBATCH --partition=debug 
#SBATCH --nodelist=gpu04
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=999:00:00


conda activate tf1
python main.py