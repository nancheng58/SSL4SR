# Recommendation System baseline Model

## Overview

This is a repo of several Recommendation System baseline.

| **Model** | article Link                     | repo link                                  | **Topic**      |
| --------- | -------------------------------- | ------------------------------------------ | -------------- |
| ASPep     | https://arxiv.org/abs/2105.00522 | https://github.com/DyGRec/ASReP            | Sequential Rec |
| DHCN      | https://arxiv.org/abs/2012.06852 | https://github.com/xiaxin1998/DHCN         | Session Rec    |
| S3Rec     | https://arxiv.org/abs/2008.07873 | https://github.com/RUCAIBox/CIKM2020-S3Rec | Sequential Rec |
| SMIN      | https://arxiv.org/abs/2110.03958 | https://github.com/SocialRecsys/SMIN       | Social Rec     |



Under each folder, there are files of each original repo and my reproduce result .

*Note: the S3Rec4MrTrmBase repe is a baseline for [MrTransformer](https://github.com/mamuyang/MrTransformer) , we get rid of the MAP and AAP loss and the hyperparameter of Pretrain and FineTune stage is distinct from S3Rec.*



## Usage

### Environments

In every baseline model folder,  if you can find the requirement.txt, you can use

`pip install -r requirements.txt`  if you use pip.

`conda install --yes --file requirements.txt`  if you use conda.

*[Note: if you run SMIN model, you can only use DGL(0.4.3), higher or lower version may cause incompatibility problem.The Anaconda channel off the shelf this version but you can find in [here](https://pypi.tuna.tsinghua.edu.cn/simple/dgl-cu102/) or install from [source](https://github.com/dmlc/dgl)*.



### Slurm

In every baseline folder, there is a slurm execute script.

About slurm usage,you can reference this link: https://slurm.schedmd.com/documentation.html

*[Note: you ought to create result folder before execute script]*

*[Note: you ought to modify  "conda activate envname" to your environment]*

```bash
#!/bin/bash
#SBATCH -e result/sas_ans_FT.err
#SBATCH -o result/sas_ans_FT.out
#SBATCH -J sas4recFT

#SBATCH --partition=debug 
#SBATCH --nodelist=gpu03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=999:00:00


conda activate torch1.8
python main.py 

```

