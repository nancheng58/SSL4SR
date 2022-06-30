# Sequential Recommendation System via Pretain

repo link: https://github.com/nancheng58/Pretraining-for-Recommender-Systems

## Overview

This is a repo of several Sequential Recommendation System baseline.

| **Model**     | Paper title and link                                         | Code link                                                    | **Topic**      | From      |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------- | --------- |
| ASReP         | *[Augmenting Sequential Recommendation with Pseudo-Prior Items via Reversely Pre-training Transformer](https://arxiv.org/abs/2105.00522)* | https://github.com/DyGRec/ASReP                              | Sequential Rec | SIGIR2021 |
| SASRec        | [Self-Attentive Sequential Recommendation](https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf) | https://github.com/kang205/SASRec                            | Sequential Rec | ICDM2018  |
| DHCN          | *[Self-Supervised Hypergraph Convolutional Networks for Session-based Recommendation](https://arxiv.org/abs/2012.06852)* | https://github.com/xiaxin1998/DHCN                           | Session Rec    | AAAI2021  |
| S3Rec         | *[S3Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization](https://arxiv.org/abs/2008.07873)* | https://github.com/RUCAIBox/CIKM2020-S3Rec                   | Sequential Rec | CIKM2020  |
| MrTransformer | *[Improving Transformer-based Sequential Recommenders through Preference Editing](https://arxiv.org/abs/2106.12120)* | https://github.com/mamuyang/MrTransformer                    | Sequential Rec | arXiv     |
| BERT4Rec      | *[BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690)* | https://github.com/FeiSun/BERT4Rec                           | Sequential Rec | CIKM2019  |
| CL4SRec       | *[Contrastive Learning for Sequential Recommendation](https://arxiv.org/abs/2010.14395)* | our reproduction via [RecBole](https://github.com/RUCAIBox/RecBole) | Sequential Rec | arXiv     |
| SGL           | *[Self-supervised Graph Learning for Recommendation](https://arxiv.org/abs/2010.10783)* | https://github.com/wujcan/SGL                                | Session Rec    | SIGIR2021 |

For CL4Rec and SGL models, we reproduce them and run experiment with [RecBole](https://github.com/RUCAIBox/RecBole).

The code is changed relative to the original code. For example, we have added the code to count the indicators of different length series in each model. In addition, for ASReP model and BERT4Rec model, we add ablation study.

## AsReP ablation study

| 消融实验    | 说明                                                         | 对应参数                                                    |
| ----------- | ------------------------------------------------------------ | ----------------------------------------------------------- |
| findByEmbed | 先计算pretrain model的item embedding，再mean pool得到seq的embedding， | --aug_mode inspire(在run_pretrain.bash文件里更改)           |
| pop         | 随机选择top50物品，再随机增广                                | 用到的函数在已写到util.py中，稍作修改即可用（需在源码更改） |
| point       | 选择top-k物品一次性生成                                      | --aug_mode seq2point(在run_pretrain.bash文件里更改)         |
| seq         | 按照自回归的方式生成                                         | --aug_mode seq2seq(在run_pretrain.bash文件里更改)           |
| -pretrain   | 去掉pretrain过程                                             | --reversed_pretrain -1(在run_finetune.bash文件里更改)       |
| -aud data   | 去掉增广数据，其实就是生成阈值M设为0                         | --M 0(在run_finetune.bash文件里更改)                        |



## Usage

### Environments

In every baseline model folder,  if you can find the requirement.txt, you can use

`pip install -r requirements.txt`  if you use pip.

`conda install --yes --file requirements.txt`  if you use conda.

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

