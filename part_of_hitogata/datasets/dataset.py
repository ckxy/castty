import torch.utils.data as data
from .bamboo import Bamboo
from .readers.builder import build_reader


class Dataset(data.Dataset):
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.use_pil = cfg.use_pil if cfg.use_pil else True

        self.reader = build_reader(cfg.reader)
        self.bamboo = Bamboo(cfg.internodes)

        self.info = self.reader.info
        forcat = self.info.pop('forcat')
        self.info.update(forcat)

        self.branch_id = 0

    def __getitem__(self, index):
        data_dict = dict(reader=self.reader, index=index, len_data_lines=len(self), branch_id=self.branch_id)
        data_dict = self.bamboo(data_dict)
        return data_dict

    def update_branch_id(self, branch_id=0):
        assert isinstance(branch_id, int) and branch_id >= 0
        self.branch_id = branch_id

    def __len__(self):
        return len(self.reader)

    def __repr__(self):
        split_str = self.bamboo.__repr__().split('\n')
        bamboo_str = split_str[0]
        for i in range(1, len(split_str)):
            bamboo_str += '\n  ' + split_str[i]

        return 'Dataset(\n  len: {}\n  reader: {}\n  bamboo: {} \n)'.format(len(self), self.reader.__repr__(), bamboo_str)
