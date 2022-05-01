# usage

[中文](./README_CN.md)

## warning
remember to comment [this line](https://github.com/Furyton/Recommender-Baseline-Model/blob/4d8831e547e3eefee36cd8ebcfb22834c31871d8/NerualNetwork/bert4rec%26sas4rec/utils.py#L75) if you are using slurm schedular

## bert4rec & sas4rec

### dataset format

the datasets are stored in the `Data` directory in text format, e.g. `ml.txt`

the index of users and items starts at 1, and the id should be both continuous.

each line indicates an interaction: user_id item_id

for any single user, the order of the interacted items should be preserved. But the order between different users could be mixed.

example
```
1 2
1 5
1 1
2 1
2 2
1 7
```

### configuration
the config file `config.json` is all you need to change

dataset part

```txt
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

```txt

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

```txt

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

this is the config you can edit for `bert4rec&sas4rec`.

### training

#### Step 1

put your dataset into the `Data` dir, make sure the format meet the requirement.

#### Step 2
edit the `config.json` file

#### Step 3

install Essential Package.
```
pip install -r requirements.txt
```

#### Step 4

run
```
python main.py
```

### note
you can also config the parameters in the command line. they will overwrite the `config.json`

example

```
python main.py --mode=test
```

you can select the config file in the command line

example

```
python main.py --config_file=config1.json

python main.py --config_file=config2.json
```

don't forget to change `test_model_path` and `resume_path` in the `config.json` when used

### explanation

after you finishing the training, you can check your directory, by default it is `experiment` 

it looks like this
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

the weight state dict is saved at `models` dir, used for test or resume training

to see the metrics changes during the training, first open cmd in the `logs` dir (Windows)

```
C:\...\logs>
```

then type the command

```
tensorboard --logdir=tensorboard_visualization
```

then it may return

```
logs>tensorboard --logdir=tensorboard_visualization
TensorFlow installation not found - running with reduced feature set.
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.4.1 at http://localhost:6006/ (Press CTRL+C to quit)
```

open `http://localhost:6006/` in the browser

![tensorboard display](tensorboard_display.PNG)

![tensorboard graph display](tensorboard_graph_display.PNG)
