#!/bin/bash
#SBATCH -e result/yelp_learning_finetune_p1.err # ��׼�����ض�����test.err�ļ�
#SBATCH -o result/yelp_learning_finetune_p1.out # ��׼����ض�����test.out�ļ�
#SBATCH -J yelp_learning_finetune_p1 # ��ҵ��ָ��Ϊbeauty_gen1

#SBATCH --partition=debug # ָ����ҵ�ύ�ķ���Ϊdebug����
#SBATCH --gres=gpu:1 # ÿ���ڵ���Ҫ����1��GPU
#SBATCH --cpus-per-task=1 # һ��������Ҫ�����CPU������Ϊ1
#SBATCH --time=999:00:00 # �������е��ʱ��Ϊ1Сʱ


conda activate python36
python3  run_yelp_learning2_loss_finetune_p1.py


