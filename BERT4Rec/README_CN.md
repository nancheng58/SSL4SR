# 使用方法

[English](./README.md)

## 注意
如果你使用slurm 来调度你的gpu资源，要注释掉[这一行](https://github.com/Furyton/Recommender-Baseline-Model/blob/4d8831e547e3eefee36cd8ebcfb22834c31871d8/NerualNetwork/bert4rec%26sas4rec/utils.py#L75)

## sas4rec

### 数据集格式

数据集放在了`Data` 目录下，文本格式，比如 `ml.txt`

user和item下标从 1 开始，必须连续

每行代表一个交互：user_id item_id

对于同一个用户，物品出现顺序不能打乱，但不同用户的数据出现位置可以混在一起

sample
```
1 2
1 5
1 1
2 1
2 2
1 7
```

### 配置
需要更改配置文件 `config.json`


dataset part

```json
load_processed_dataset: the dataset you put in the `Data` will be processed into .pkl, 
                        you can load it for saving time. true or false, e.g. false
processed_dataset_path: absolute path, e.g. "C:processed/ml.pkl"

dataloader_random_seed: float, default=0.0

train_batch_size: batch size, e.g. 64
val_batch_size: batch size, e.g. 64
test_batch_size: batch size, e.g. 64


prop_sliding_window: propotion of the sliding window step, if the input seq is exceeding the max_len, 
                     you can use a sliding window to generate a sequence of input. default: 0.1,  
                     if you don't want this sliding windown, set this parameter as -1.0.

worker_number: for multi-processer, usually it could be 4 times #cpu core you have


train_negative_sampler_code: popular or random, e.g. "popular"
train_negative_sample_size: for bert and sas, this is unused, set as 0
train_negative_sampling_seed: 0


test_negative_sampler_code: popular or random, e.g. "popular"
test_negative_sample_size: default: 100
test_negative_sampling_seed: 0

```

training part

```json

mode: train or test, e.g. "train"

test_model_path: absolute path, e.g. "C://model//my_model.pth"
resume_path: absolute path, e.g. "C://model//check_point_model.pth"

device: cpu or cuda, default: "cpu"
num_gpu: default: 1
device_idx: note: this is a string, you can type "0", or "0, 1, 2"


optimizer: "Adam" or "SGD"
lr: learning rate, e.g. 0.001
weight_decay: l2 regularization, e.g. 0.01

decay_step: decay step for StepLR, e.g. 15
gamma: Gamma for StepLR, e.g. 0.1


num_epochs: number of epochs for training, e.g. 100

log_period_as_iter: after every certain number of iterations, the model weight will be saved as checkpoint

metric_ks: ks for Metric@k, there are 3 types of metric: MRR, NDCG and HIT, e.g. [10, 20, 50]
best_metric: Metric for determining the best model, e.g. "NDCG@10"

show_process_bar: show the processing bar when training or testing, true or false, e.g.false

```

model part

```json

model_code: bert or sas

FOR BERT

bert_max_len: Length of sequence for bert, e.g. 200
bert_hidden_units: Size of hidden vectors, e.g. 64
bert_num_blocks: number of transformer layers, e.g. 2
bert_num_heads: number of heads for multi-attention, e.g. 2
bert_dropout: used for transformer blocks
bert_hidden_dropout: used for hidden unit layers

FOR SAS

sas_max_len : Length of sequence for sas, e.g. 200
sas_hidden_units: Size of hidden vectors, e.g. 64
sas_num_blocks: number of transformer layers, e.g. 2
sas_heads: number of heads for multi-attention, e.g. 2
sas_dropout: dropout rate, e.g. 0.2
l2_emb: normalization, e.g. 0.0

experiment_dir: where you want to put you trained model and log info in, e.g. "experiments"
experiment_description: default: "test"
dataset_name: the dataset filename you put in the `Data` dir, e.g. "ml.txt"
```


### 训练

#### Step 1
将数据集放到 `Data` 目录下，确保格式正确

#### Step 2
编辑 `config.json` 文件

#### Step 3
安装库

```
pip install -r requirements.txt
```

#### Step 4
run
```
python main.py
```

### note

可以使用命令行参数，它会覆盖 `config.json` 

example

```
python main.py --mode=test
```

命令行参数中可以选择配置文件


example

```
python main.py --config_file=config1.json

python main.py --config_file=config2.json
```

别忘记更改 `test_mdoel_path` 或者 `resume_path` 如果需要使用 mode=test 或者 resume training

### 解释

训练结束后，在对应目录下可以找到模型参数和结果，默认放在 `experiment` 目录下，可以通过 `config.json` 中的 `experiment_dir` 中更改

目录结构如下
```
- test_bert_2021-03-28_0
    - logs
        - tensorboard_visualization
        test_metrics.json
     - models
        best_acc_model.pth
        checkpoint-recent.pth
        checkpoint-recent.pth.final
     config.json
  
- test_bert_2021-03-28_1

```

models 里保存模型权重信息，用来test 或者 resume training

可视化或者用图表展示模型训练过程，首先在 `logs` 目录下打开 cmd （windows 下）

```
tensorboard --logdir=tensorboard_visualization
```

打开浏览器进入`http://localhost:6006/`

![tensorboard display](tensorboard_display.PNG)

![tensorboard graph display](tensorboard_graph_display.PNG)
