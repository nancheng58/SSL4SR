#!/bin/bash
#SBATCH -e result/pre_ASReP_B_M20_gen15_inspire_findKByPool.err
#SBATCH -o result/pre_ASReP_B_M20_gen15_inspire_findKByPool.out
#SBATCH -J pre_B_ASReP

#SBATCH --partition=edu 
#SBATCH --nodelist=gpu02
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=999:00:00


conda activate tf1
python main.py --dataset=Beauty --train_dir=default --lr=0.001 --hidden_units=128 --maxlen=50 --dropout_rate=0.5 --num_blocks=2 --l2_emb=0.0 --num_heads=2 --evalnegsample 99 --reversed 1 --reversed_gen_num 20 --M 20 --aug_mode inspire
#python main.py --dataset=ML-1M --train_dir=default --lr=0.001 --hidden_units=128 --maxlen=50 --dropout_rate=0.5 --num_blocks=2 --l2_emb=0.0 --num_heads=4 --evalnegsample 99 --reversed 1 --reversed_gen_num 10 --M 50
#python main.py --dataset=yelp --train_dir=default --lr=0.001 --hidden_units=64 --maxlen=50 --dropout_rate=0.5 --num_blocks=1 --l2_emb=0.0 --num_heads=2 --evalnegsample 99 --reversed 1 --reversed_gen_num 20 --M 20 --aug_mode seq2point