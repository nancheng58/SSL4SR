#!/bin/bash
#SBATCH -e result/ASReP_B_gen15_M20_inspire_findKByPool-pretrain.err
#SBATCH -o result/ASReP_B_gen15_M20_inspire_findKByPool=pretrain.out
#SBATCH -J ASReP_B

#SBATCH --partition=edu
#SBATCH --nodelist=gpu02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=999:00:00


conda activate tf1
python main.py --dataset=Beauty --train_dir=default --lr=0.001 --hidden_units=128 --maxlen=50 --dropout_rate=0.5 --num_blocks=2 --l2_emb=0.0 --num_heads=2 --evalnegsample 99 --reversed_pretrain -1 --aug_traindata 15 --M 20
#python main.py --dataset=ML-1M --train_dir=default --lr=0.001 --hidden_units=128 --maxlen=50 --dropout_rate=0.5 --num_blocks=2 --l2_emb=0.0 --num_heads=4 --evalnegsample 99 --reversed_pretrain 1 --aug_traindata 10 --M 50
#python main.py --dataset=yelp --train_dir=default --lr=0.001 --hidden_units=64 --maxlen=50 --dropout_rate=0.5 --num_blocks=1 --l2_emb=0.0 --num_heads=2 --evalnegsample 99 --reversed_pretrain 1 --aug_traindata 15 --M 18 --aug_mode seq2point