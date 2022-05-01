from .popular import PopularNegativeSampler
from .random import RandomNegativeSampler

from copy import deepcopy

# negative samplers for inference
NEGATIVE_SAMPLERS = {
    PopularNegativeSampler.code(): PopularNegativeSampler,
    RandomNegativeSampler.code(): RandomNegativeSampler,
}


def negative_sampler_factory(code, train, val, test, user_count, item_count, sample_size, seed, save_folder, dataset_name):
    negative_sampler = NEGATIVE_SAMPLERS[code]

    return negative_sampler(deepcopy(train), deepcopy(val), deepcopy(test),
                            user_count, item_count, sample_size, seed, save_folder, dataset_name)
