# Sequential Recommendation System via Self-supervised Learning

## Overview
This repo is the code of our survey paper "[基于自监督的预训练在推荐系统中的研究综述](https://github.com/nancheng58/Self-supervised-learning-for-Sequential-Recommender-Systems/blob/main/%E5%9F%BA%E4%BA%8E%E8%87%AA%E7%9B%91%E7%9D%A3%E7%9A%84%E9%A2%84%E8%AE%AD%E7%BB%83%E5%9C%A8%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E4%B8%AD%E7%9A%84%E7%A0%94%E7%A9%B6%E7%BB%BC%E8%BF%B0.pdf)" accepted by CCIR 2023, which collects several codes and datasets of Self-supervised learning Sequential Recommendation System baselines.

| **Model**     | Paper title and link                                         | Code link                                                    | **Topic**      | From      |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------- | --------- |
| ASReP         | *[Augmenting Sequential Recommendation with Pseudo-Prior Items via Reversely Pre-training Transformer](https://arxiv.org/abs/2105.00522)* | https://github.com/DyGRec/ASReP                              | Sequential Rec | SIGIR2021 |
| SASRec        | [Self-Attentive Sequential Recommendation](https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf) | https://github.com/kang205/SASRec                            | Sequential Rec | ICDM2018  |
| DHCN          | *[Self-Supervised Hypergraph Convolutional Networks for Session-based Recommendation](https://arxiv.org/abs/2012.06852)* | https://github.com/xiaxin1998/DHCN                           | Session Rec    | AAAI2021  |
| S3Rec         | *[S3Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization](https://arxiv.org/abs/2008.07873)* | https://github.com/RUCAIBox/CIKM2020-S3Rec                   | Sequential Rec | CIKM2020  |
| MrTransformer | *[Improving Transformer-based Sequential Recommenders through Preference Editing](https://arxiv.org/abs/2106.12120)* | https://github.com/mamuyang/MrTransformer                    | Sequential Rec | TOIS2022     |
| BERT4Rec      | *[BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690)* | https://github.com/FeiSun/BERT4Rec                           | Sequential Rec | CIKM2019  |
| CL4SRec       | *[Contrastive Learning for Sequential Recommendation](https://arxiv.org/abs/2010.14395)* | our reproduction via [RecBole](https://github.com/RUCAIBox/RecBole) and [DuoRec](https://github.com/RuihongQiu/DuoRec)| Sequential Rec | ICDE2022     |
| SGL           | *[Self-supervised Graph Learning for Recommendation](https://arxiv.org/abs/2010.10783)* | https://github.com/wujcan/SGL                                | Session Rec    | SIGIR2021 |

For CL4Rec and SGL models, we reproduce them and run experiment with [RecBole](https://github.com/RUCAIBox/RecBole).

The code is changed relative to the original code. For example, we have added the code to count the indicators of different length series in each model.

## Datasets


|                       | Beauty      | ML-1M      | Yelp        |
| --------------------- | ----------- | ---------- | ----------- |
| User                  | 22364       | 6040       | 22845       |
| Item                  | 12102       | 3352       | 16552       |
| Interaction           | 194687      | 269721     | 237004      |
| Total File            | 4.18M       | 5.30M      | 5.19M       |
| Min_len               | 5           | 17         | 5           |
| Max_len               | 50          | 50         | 50          |
| Avg_len               | 8.7057      | 44.6557    | 10.37443    |
| Density               | 0.07194251% | 1.3322134% | 0.06267784% |
| Attributes            | 2320        | 18         | 1158        |
| Min. Attribute / Item | 1           | 1          | 0           |
| Max. Attribute / Item | 9           | 6          | 33          |
| Avg. Attribute / Item | 3.9391      | 1.7072     | 4.9205      |

| length  | Beauty            | ML-1M            | Yelp              |
| ------- | ----------------- | ---------------- | ----------------- |
| [0,20)  | 21228 \| 94.9202% | 177 \| 2.9305%   | 20744 \| 90.8032% |
| [20,30) | 655 \| 2.9289%    | 684 \| 11.3245%  | 1094 \| 4.7888%   |
| [30,40) | 231 \| 1.0330%    | 543 \| 8.9901%   | 511 \| 2.2368%    |
| [40,50] | 250 \| 1.1179%    | 4636 \| 76.7550% | 496 \| 2.1712%    |
| overall | 22364 \| 100%     | 6040 \| 100%     | 22845 \| 100%     |

### Datasets PreProcessing
We refer to the method in [1,2,3] to process the datasets. If the user interacts with the item, we will convert the interaction with a clear score into implicit positive feedback. After that, we will group the interactive information according to users. We will sort the items for each user according to the timestamp of their interaction with the items. Because this work aims not to investigate the "cold start" issue in the recommendation system, we circularly filter out users with less than 5 interactions and items with less than 5 interactions. In addition, there are users with too much interactive data in the dataset used in this work, so we limit the maximum length of the user interaction sequences to 50. Because the yelp dataset is too large, we adopted a processing method similar to [3], and only the 2019 data of the dataset was intercepted. 


ref: 
[1] Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In ICDM. IEEE, 197–206.

[2] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In CIKM. 1441–1450.

[3] S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management (CIKM '20)
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
#SBATCH --nodelist=gpuxx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=999:00:00


conda activate xxx
python main.py 

```

