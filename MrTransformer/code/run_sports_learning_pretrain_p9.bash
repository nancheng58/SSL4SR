#!/bin/bash
#SBATCH -e result/sports_learning_pretrain_p9.err # ��׼�����ض�����test.err�ļ�
#SBATCH -o result/sports_learning_pretrain_p9.out # ��׼����ض�����test.out�ļ�
#SBATCH -J sports_learning_pretrain_p9 # ��ҵ��ָ��Ϊbeauty_gen1

#SBATCH --partition=debug # ָ����ҵ�ύ�ķ���Ϊdebug����
#SBATCH --gres=gpu:1 # ÿ���ڵ���Ҫ����1��GPU
#SBATCH --cpus-per-task=1 # һ��������Ҫ�����CPU������Ϊ1
#SBATCH --time=999:00:00 # �������е��ʱ��Ϊ1Сʱ


conda activate python36
python3  run_sports_learning2_loss_pretrain_p9.py


