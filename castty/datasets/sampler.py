import torch
from .readers.cat import CatReader
from ..utils.registry import Registry, build_from_cfg
from torch.utils.data.sampler import WeightedRandomSampler


SAMPLER = Registry('sampler')


def build_sampler(cfg, **default_args):
    return build_from_cfg(cfg, SAMPLER, default_args)


@SAMPLER.register_module()
class CustomWeightedRandomSampler(WeightedRandomSampler):
    def __init__(self, dataset, weights, num_samples=None, **kwargs):
        reader = dataset.reader

        assert isinstance(reader, CatReader)
        for w in weights:
            assert w > 0

        groups = reader.groups

        assert len(weights) + 1 == len(groups)

        intl_weights = torch.FloatTensor(groups[-1]).fill_(1)
        for i in range(1, len(groups)):
            intl_weights[groups[i - 1]:groups[i]] = weights[i - 1]

        super(CustomWeightedRandomSampler, self).__init__(intl_weights, len(intl_weights) if num_samples is None else num_samples)
