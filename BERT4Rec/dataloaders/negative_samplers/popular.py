from .base import AbstractNegativeSampler

from tqdm import trange

from collections import Counter

import numpy as np


class PopularNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'popular'

    def generate_negative_samples(self):
        popular_items = self.items_by_popularity()

        keys = np.array([x for x in popular_items.keys()])
        values = popular_items.values()
        sum_value = np.sum([x for x in values])
        probability = [value / sum_value for value in values]

        negative_samples = {}
        print('Sampling negative items')
        for user in trange(0, self.user_count):
            seen = set(self.train[user])
            seen.update(self.val[user])
            seen.update(self.test[user])

            samples = []
            while len(samples) < self.sample_size:
                sampled_ids = np.random.choice(keys, self.sample_size, replace=False, p=probability)
                sampled_ids = [x for x in sampled_ids if x not in seen and x not in samples]
                samples.extend(sampled_ids[:])
            # for item in popular_items:
            #     if len(samples) == self.sample_size:
            #         break
            #     if item in seen:
            #         continue
            #     samples.append(item)

            negative_samples[user] = samples[:self.sample_size]

        return negative_samples

    def items_by_popularity(self):
        popularity = Counter()
        for user in range(0, self.user_count):
            popularity.update(self.train[user])
            popularity.update(self.val[user])
            popularity.update(self.test[user])
        # popular_items = sorted(popularity, key=popularity.get, reverse=True)
        return popularity
