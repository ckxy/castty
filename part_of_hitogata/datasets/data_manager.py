import math
import importlib
import torch.utils.data
from copy import deepcopy
from .dataset import Dataset
from .collator import Collator
from .sampler import build_sampler
from .batch_sampler import build_batch_sampler


class DataManager(object):
    def __init__(self, cfg):
        self.cfg = cfg.data_loader
        drop_uneven = self.cfg.drop_uneven if self.cfg.drop_uneven else False

        self.dataset = Dataset(cfg.dataset)

        if self.cfg.sampler:
            sampler_cfg = deepcopy(self.cfg.sampler)
            sampler_cfg.update(dict(dataset=self.dataset))
            sampler = build_sampler(sampler_cfg)
        else:
            if self.cfg.serial_batches:
                sampler = torch.utils.data.sampler.SequentialSampler(self.dataset)
            else:
                sampler = torch.utils.data.sampler.RandomSampler(self.dataset)

        if self.cfg.batch_sampler:
            sampler_cfg = deepcopy(self.cfg.batch_sampler)
            sampler_cfg.update(dict(reader=self.dataset.reader, sampler=sampler, batch_size=self.cfg.batch_size, drop_uneven=drop_uneven))
            batch_sampler = build_batch_sampler(sampler_cfg)
        else:
            batch_sampler = torch.utils.data.BatchSampler(sampler, self.cfg.batch_size, drop_uneven)

        if self.cfg.collator:
            self.cfm = Collator(self.cfg.collator)
            self.cf = self.cfm.collate_fn
        else:
            self.cf = None

        self.info = self.dataset.info
        self.oobmab = self.dataset.bamboo.reverse

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            num_workers=self.cfg.num_threads,
            pin_memory=self.cfg.pin_memory,
            collate_fn=self.cf,
            batch_sampler=batch_sampler,
        )

    def __repr__(self):
        split_str = self.dataset.__repr__().split('\n')
        dataset_str = split_str[0]
        for i in range(1, len(split_str)):
            dataset_str += '\n  ' + split_str[i]

        if self.cf:
            # cf_str = self.cfm.__repr__()
            split_str = self.cfm.__repr__().split('\n')
            cf_str = split_str[0]
            for i in range(1, len(split_str)):
                cf_str += '\n  ' + split_str[i]
        else:
            cf_str = 'None'
        return 'Dataloader(\n  batch_size: {}\n  len: {}\n  serial_batches: {}  \n  ' \
               'collate_function: {} \n  dataset: {}\n)'.format(self.cfg.batch_size, len(self),
                                                                self.cfg.serial_batches, cf_str, dataset_str)

    def __len__(self):
        return len(self.dataloader)

    def load_data(self):
        return self.dataloader
