import math
import importlib
import torch.utils.data
from .collator import Collator
from .dataset import Dataset


def find_sampler_using_name(sampler_mode):
    sampler_lib = importlib.import_module('datasetsnx.sampler')

    sampler = None
    for name, cls in sampler_lib.__dict__.items():
        if name.lower() == sampler_mode.lower():
            sampler = cls

    if sampler is None:
        raise NotImplementedError(
            "sampler.py中，没有以{}的小写为名的类".format(sampler_mode))

    return sampler


def find_analyser_using_name(sampler_mode):
    analyser_lib = importlib.import_module('datasetsnx.analyser')

    analyser = None
    for name, cls in analyser_lib.__dict__.items():
        if name.lower() == sampler_mode.lower():
            analyser = cls

    if analyser is None:
        raise NotImplementedError(
            "analyser.py中，没有以{}的小写为名的类".format(sampler_mode))

    return analyser


def create_data_manager(cfg):
    data_manager = DataManager(cfg)
    print(data_manager)
    return data_manager
    # return data_loader.load_data(), data_loader.info


# class InfoManager(object):
#     def __init__(self, dataset):
#         self.dataset = dataset

#     def __iter__(self):
#         return self

#     def __next__(self):
#         if self.count >= len(self):
#             self.count = 0
#             if not self.cfg.serial_batches:
#                 random.shuffle(self.indices)
#             raise StopIteration
#         else:
#             if len(self) - self.count == 1:
#                 batch_indices = self.indices[self.count * self.batch_size:]
#             else:
#                 batch_indices = self.indices[
#                                 self.count * self.batch_size: (self.count + 1) * self.batch_size]

#             res = self.__processing(batch_indices)
#             self.count += 1

#             return res

#     def __len__(self):
#         return len(self.dataset)


class DataManager(object):
    def __init__(self, cfg):
        self.cfg = cfg.data_loader
        drop_uneven = self.cfg.drop_uneven if self.cfg.drop_uneven else False

        # self.dataset = find_dataset_using_name(cfg.dataset.name)(cfg.dataset)
        self.dataset = Dataset(cfg.dataset)

        analysis = None
        if self.cfg.analyser:
            print('processing {}'.format(self.cfg.analyser[0]))
            analysis = find_analyser_using_name(self.cfg.analyser[0])(self.dataset, **self.cfg.analyser[1])
        else:
            analysis = None

        if self.cfg.batch_sampler:
            if self.cfg.serial_batches:
                sampler = torch.utils.data.sampler.SequentialSampler(self.dataset)
            else:
                sampler = torch.utils.data.sampler.RandomSampler(self.dataset)

            batch_sampler = find_sampler_using_name(self.cfg.batch_sampler[0])(sampler=sampler, batch_size=self.cfg.batch_size, drop_uneven=self.cfg.drop_uneven, analysis=analysis, **self.cfg.batch_sampler[1])

        if self.cfg.collator:
            self.cfm = Collator(self.cfg.collator)
            self.cf = self.cfm.collate_fn
        else:
            self.cf = None

        self.info = self.dataset.info
        self.info['oobmab'] = self.dataset.bamboo.reverse
        # print(self.dataset.bamboo.rper)
        # exit()

        if self.cfg.batch_sampler:
            self.dataloader = torch.utils.data.DataLoader(
                    self.dataset,
                    num_workers=self.cfg.num_threads,
                    pin_memory=self.cfg.pin_memory,
                    collate_fn=self.cf,
                    batch_sampler=batch_sampler,
                )
        else:
            self.dataloader = torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=self.cfg.batch_size,
                    shuffle=not self.cfg.serial_batches,
                    num_workers=self.cfg.num_threads,
                    pin_memory=self.cfg.pin_memory,
                    drop_last=drop_uneven,
                    collate_fn=self.cf,
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
        # return math.ceil(1.0 * len(self.dataset) / self.cfg.batch_size)
        return len(self.dataloader)

    def load_data(self):
        return self.dataloader
