#!/bin/bash
#SBATCH -e result/toys_learning_finetune_p11.err # ��׼�����ض�����test.err�ļ�
#SBATCH -o result/toys_learning_finetune_p11.out # ��׼����ض�����test.out�ļ�
#SBATCH -J toys_learning_finetune_p11 # ��ҵ��ָ��Ϊbeauty_gen1

#SBATCH --partition=debug # ָ����ҵ�ύ�ķ���Ϊdebug����
#SBATCH --nodelist=gpu04
#SBATCH --gres=gpu:1 # ÿ���ڵ���Ҫ����1��GPU
#SBATCH --cpus-per-task=1 # һ��������Ҫ�����CPU������Ϊ1
#SBATCH --time=999:00:00 # �������е��ʱ��Ϊ1Сʱ


conda activate python36
python3  run_toys_learning2_loss_finetune_p11.py


