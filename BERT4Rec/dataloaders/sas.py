from .base import AbstractDataloader

import torch
import torch.utils.data as data_utils
from copy import deepcopy
from multiprocessing import Process, Queue
from collections import defaultdict, Counter
import numpy as np


class SASDataLoader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        self.max_len = args.max_len

    @classmethod
    def code(cls):
        return 'sas'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataloader = WarpSampler(user_train=self.train, item_num=self.item_count, batch_size=self.args.train_batch_size,
                                 max_len=self.max_len, device=self.args.device, num_workers=self.worker_num)

        return dataloader

    # def _get_train_dataset(self):
    #     dataset = SASTrainDataset(user_train=self.train, user_num=self.user_count, item_num=self.item_count,
    #                               max_len=self.max_len, rng=self.rng)
    #
    #     return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True, num_workers=self.worker_num)
        return dataloader

    def _get_eval_dataset(self, mode):
        answers = self.val if mode == 'val' else self.test
        # train_dataset = None
        if mode == 'val':
            train_dataset = deepcopy(self.train)
        else:
            train_dataset = deepcopy(self.train)
            for index, seq in enumerate(train_dataset):
                seq.append(self.val[index][0])

        dataset = SASEvalDataset(train_dataset, answers, self.max_len, self.test_negative_samples)
        return dataset


def random_neq(l, r, exclusive: set, size):
    a = list(set(range(l, r + 1)) - exclusive)
    return [a[i] for i in np.random.randint(0, len(a), size=size)]


def sample_function(user_train, item_num, batch_size, max_len, result_queue):
    def sample():
        train = user_train[np.random.randint(0, len(user_train))][-max_len:]
        padding_len = max_len - len(train) + 1

        seq = padding_len * [0] + train[:-1]
        pos = padding_len * [0] + train[1:]
        neg = padding_len * [0] + random_neq(l=0, r=item_num, exclusive=set(train), size=len(train) - 1)

        return seq, pos, neg

    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, user_train, item_num, batch_size, max_len, device, num_workers=1):
        self.cnt = 0
        self.num_batch = len(user_train) // batch_size
        self.result_queue = Queue(maxsize=num_workers * 10)
        self.processors = []
        self.device = device
        for i in range(num_workers):
            self.processors.append(
                Process(target=sample_function, args=(
                    user_train, item_num, batch_size, max_len, self.result_queue
                ))
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def __iter__(self):
        self.cnt = 0
        return self

    def __next__(self):
        if self.cnt < self.num_batch:
            self.cnt += 1
            return self.result_queue.get()
        else:
            raise StopIteration

    def __len__(self):
        return self.num_batch

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


class SASEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, negative_samples):
        self.u2seq = u2seq
        self.users = range(len(u2seq))
        self.u2answer = u2answer
        self.max_len = max_len
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = deepcopy(self.u2seq[user])
        answer = deepcopy(self.u2answer[user])

        if user == 959:
            print('here')

        negs = deepcopy(self.negative_samples[user])

        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return np.array(seq, dtype=np.int32), np.array(candidates, dtype=np.int32), torch.LongTensor(labels)
