from collections import defaultdict

from .bert import BertDataloader
from .sas import SASDataLoader

import _pickle as cPickle
import os
from os import path


DATALOADERS = {
    BertDataloader.code(): BertDataloader,
    SASDataLoader.code(): SASDataLoader
}


def data_partition(fname, max_len, prop_sliding_window):
    usernum = 0
    itemnum = 0
    max_len = 50
    User = defaultdict(list)
    user_train = []
    user_valid = []
    user_test = []
    # assume user/item index starting from 1

    current_directory = path.dirname(__file__)
    parent_directory = path.split(current_directory)[0]
    dataset_filepath = path.join(parent_directory, 'Data', fname)
    train_dataset_filepath = path.join(dataset_filepath, "train.txt")
    valid_dataset_filepath = path.join(dataset_filepath, "valid.txt")
    test_dataset_filepath = path.join(dataset_filepath, "test.txt")
    f = open(train_dataset_filepath, 'r')
    # traindata = []
    traindata = defaultdict(list)
    for line in f:
        u, i, _ = line.rstrip().split('\t')
        u = int(u)+1
        i = int(i)+1
        itemnum = max(i, itemnum)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        traindata[u].append(i)
    f = open(valid_dataset_filepath, 'r')
    for line in f:
        u, i,_  = line.rstrip().split('\t')
        u = int(u)+1
        i = int(i)+1
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        user_valid.append([i])
    f = open(test_dataset_filepath, 'r')
    for line in f:
        u, i,_  = line.rstrip().split('\t')
        u = int(u)+1
        i = int(i)+1
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        user_test.append([i])
    # data = []
    # sliding window
    # sliding_step = int(prop_sliding_window * max_len) if prop_sliding_window != -1.0 else max_len

    # for user in User:
    #     if len(User[user]) < 3:
    #         continue
    #     if len(User[user]) <= max_len:
    #         data.append(User[user])
    #     else:
    #         # print("FILM LOVER: ", user)
    #         beg_idx = range(len(User[user]) - max_len, 0, -sliding_step)
    #         # beg_idx.append(0)
    #         for i in beg_idx[::-1]:
    #             data.append(User[user][i:i + max_len])

    # usernum = len(data)

    for i in range(1, usernum+1):
        user_train.append(traindata[i])
    #     # user_valid[i] = []
    #     user_valid.append([data[i][-2]])
    #     # user_test[i] = []
    #     user_test.append([data[i][-1]])

    return [user_train, user_valid, user_test, usernum, itemnum]


def dataloader_factory(args):
    if args.load_processed_dataset:
        dataset = cPickle.load(open(path.normpath(args.processed_dataset_path), 'rb'))
    else:
        dataset = data_partition(args.data_name, args.max_len, args.prop_sliding_window)

        current_directory = path.dirname(__file__)
        parent_directory = path.split(current_directory)[0]

        processed_dir = path.join(parent_directory, 'Data', 'Processed')

        if not path.exists(processed_dir):
            os.makedirs(processed_dir)

        dataset_filepath = path.join(processed_dir, args.data_name + '_processed.p')

        cPickle.dump(dataset, open(dataset_filepath, 'wb'))
    dataloader = DATALOADERS[args.model_code]
    dataloader = dataloader(args, dataset)
    train, val, test = dataloader.get_pytorch_dataloaders()
    return train, val, test
