#!/bin/bash
#SBATCH -e result/diginetica.err
#SBATCH -o result/diginetica.out
#SBATCH -J DHCN_diginetica

#SBATCH --partition=debug 
#SBATCH --nodelist=gpu03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=999:00:00


python main.py