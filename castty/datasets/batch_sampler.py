import torch
import itertools
from copy import deepcopy
from ..utils.registry import Registry, build_from_cfg
from torch.utils.data.sampler import BatchSampler, Sampler


BATCHSAMPLER = Registry('batch_sampler')


def build_batch_sampler(cfg, **default_args):
    return build_from_cfg(cfg, BATCHSAMPLER, default_args)


@BATCHSAMPLER.register_module()
class StepsBatchSampler(BatchSampler):
    def __init__(self, steps, sampler, batch_size=1, **kwargs):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )

        assert steps > 0

        self.sampler = sampler
        self.batch_size = batch_size
        self.steps = steps

    def __iter__(self):
        batch = []
        for i in range(self.steps):
            sampler = deepcopy(self.sampler)
            for idx in sampler:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                    break

    def __len__(self):
        return self.steps


if __name__ == '__main__':
    from torch.utils.data.sampler import SequentialSampler, RandomSampler
    a = list(StepsSampler(RandomSampler(range(6)), batch_size=4, steps=10))
    print(a)
