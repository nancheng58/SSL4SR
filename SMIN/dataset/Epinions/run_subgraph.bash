#!/bin/bash
#SBATCH -e result.err
#SBATCH -o result.out
#SBATCH -J SMIN_Epinion_Gen

#SBATCH --partition=debug 
#SBATCH --nodelist=gpu05
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --time=999:00:00


python GenerateSubGraph.py