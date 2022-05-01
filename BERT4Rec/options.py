from dataloaders import DATALOADERS
from models import MODELS
from trainers import TRAINERS
import json

import argparse


parser = argparse.ArgumentParser(description='RecPlay')

################
# Top Level
################
parser.add_argument('--config_file', type=str, default='config.json')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'resume'])
parser.add_argument('--load_processed_dataset', type=bool, default=False)
parser.add_argument('--processed_dataset_path', type=str, default=None)
################
# Test
################
parser.add_argument('--test_model_path', type=str, default=None)

parser.add_argument('--resume_path', type=str, default=None)

################
# Dataset
################
# parser.add_argument('--min_rating', type=int, default=4, help='Only keep ratings greater than equal to this value')
# parser.add_argument('--min_uc', type=int, default=5, help='Only keep users with more than min_uc ratings')
# parser.add_argument('--min_sc', type=int, default=0, help='Only keep items with more than min_sc ratings')
parser.add_argument('--split', type=str, default='leave_one_out', help='How to split the datasets')

################
# Dataloader
################
parser.add_argument('--dataloader_random_seed', type=float, default=0.0)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--prop_sliding_window', type=float, default=0.1)
parser.add_argument('--worker_number', type=int, default=1)
################
# NegativeSampler
################
parser.add_argument('--train_negative_sampler_code', type=str, default='random', choices=['popular', 'random'],
                    help='Method to sample negative items for training. Not used in bert')
parser.add_argument('--train_negative_sample_size', type=int, default=100)
parser.add_argument('--train_negative_sampling_seed', type=int, default=None)
parser.add_argument('--test_negative_sampler_code', type=str, default='random', choices=['popular', 'random'],
                    help='Method to sample negative items for evaluation')
parser.add_argument('--test_negative_sample_size', type=int, default=100)
parser.add_argument('--test_negative_sampling_seed', type=int, default=None)

################
# Trainer
################
# device #
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--device_idx', type=str, default='0') # [0, 1, 2 ... ]
# optimizer #
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
# lr scheduler #
parser.add_argument('--decay_step', type=int, default=15, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
# epochs #
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
# logger #
parser.add_argument('--log_period_as_iter', type=int, default=12800)
# evaluation #
parser.add_argument('--metric_ks', nargs='+', type=int, default=[10, 20, 50], help='ks for Metric@k')
parser.add_argument('--best_metric', type=str, default='NDCG@10', help='Metric for determining the best model')
# processing #
parser.add_argument('--show_process_bar', type=bool, default=False, help='show the processing bar or not')
################
# Model
################
parser.add_argument('--model_code', type=str, default='bert', choices=MODELS.keys())
parser.add_argument('--model_init_seed', type=int, default=None)
parser.add_argument('--max_len', type=int, default=50, help='Length of sequence')
# BERT #
parser.add_argument('--bert_hidden_units', type=int, default=None, help='Size of hidden vectors (d_model)')
parser.add_argument('--bert_num_blocks', type=int, default=None, help='Number of transformer layers')
parser.add_argument('--bert_num_heads', type=int, default=None, help='Number of heads for multi-attention')
parser.add_argument('--bert_dropout', type=float, default=None, help='Dropout probability to use throughout the model')
parser.add_argument('--bert_mask_prob', type=float, default=None, help='Probability for masking items in the training sequence')
parser.add_argument('--bert_hidden_dropout', type=float, default=None)

# SAS #
parser.add_argument('--sas_hidden_units', type=int, default=64, help='Size of hidden vectors')
parser.add_argument('--sas_num_blocks', type=int, default=2, help='Number of transformer layers')
parser.add_argument('--sas_heads', type=int, default=2, help='Number of heads for multi-attention')
parser.add_argument('--sas_dropout', type=float, default=0.2, help='Dropout probability to use throughout the model')
parser.add_argument('--l2_emb', type=float, default=0.0)
################
# Experiment
################
parser.add_argument('--experiment_dir', type=str, default='experiments')
parser.add_argument('--experiment_description', type=str, default='test')
parser.add_argument('--dataset_name', type=str, default=None)
parser.add_argument('--num_items', type=int, default=None, help='Number of total items')
################
args = parser.parse_args()

