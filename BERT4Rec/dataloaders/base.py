from .negative_samplers import negative_sampler_factory

from abc import *
from numpy import random
import copy


# [user_train, user_valid, user_test, usernum, itemnum]


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset):
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random  # rng.random(l, r) -> rand int \in [l, r - 1]
        save_folder = ''
        self.train = dataset[0]
        self.val = dataset[1]
        self.test = dataset[2]
        self.user_count = dataset[3]
        self.item_count = dataset[4]
        args.num_items = self.item_count
        self.worker_num = args.worker_number

        code = args.train_negative_sampler_code
        train_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                          self.user_count, self.item_count,
                                                          args.train_negative_sample_size,
                                                          args.train_negative_sampling_seed,
                                                          save_folder, args.data_name)
        code = args.test_negative_sampler_code
        test_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         save_folder, args.data_name)

        self.train_negative_samples = train_negative_sampler.get_negative_samples()
        self.test_negative_samples = test_negative_sampler.get_negative_samples()

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def get_pytorch_dataloaders(self):
        pass
