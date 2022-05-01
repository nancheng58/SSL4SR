#!/bin/bash
#SBATCH -e result/new_nextitem_prediction_cell.err
#SBATCH -o result/new_nextitem_prediction_cell.out
#SBATCH -J modify_ASReP_next_prediction

#SBATCH --partition=debug 
#SBATCH --nodelist=gpu04
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=999:00:00


conda activate tf1
python main.py --dataset=Cell_Phones_and_Accessories --train_dir=default --lr=0.001 --hidden_units=32 --maxlen=100 --dropout_rate=0.5 --num_blocks=2 --l2_emb=0.0 --num_heads=2 --evalnegsample 100 --reversed_pretrain 1  --aug_traindata 17 --M 18
