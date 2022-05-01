#!/bin/bash
#SBATCH -e result/gen_sports_p9.err # 标准错误重定向至test.err文件
#SBATCH -o result/gen_sports_p9.out # 标准输出重定向至test.out文件
#SBATCH -J gen_sports_p9 # 作业名指定为beauty_gen1

#SBATCH --partition=debug # 指定作业提交的分区为debug分区
#SBATCH --cpus-per-task=1 # 一个任务需要分配的CPU核心数为1
#SBATCH --time=999:00:00 # 任务运行的最长时间为1小时

conda activate python36
python3  gen_data_sports_learning_faiss_p9.py


