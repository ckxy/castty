import torch.utils.data as data
from .bamboo import Bamboo
from .readers.builder import build_reader


class Dataset(data.Dataset):
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.use_pil = cfg.use_pil if cfg.use_pil else True

        self.reader = build_reader(cfg.reader)
        self.bamboo = Bamboo(cfg.internodes)

        self.data_lines, self.info = self.reader.get_dataset_info()
        self.get_data_info_fn = self.reader.get_data_info

        self.branch_id = 0
        
        if 0 < self.cfg.max_size < len(self.data_lines):
            self.data_lines = self.data_lines[:self.cfg.max_size]

    def __getitem__(self, index):
        data_dict = dict(reader=self.reader, index=index, len_data_lines=len(self), branch_id=self.branch_id)
        data_dict = self.bamboo(data_dict)
        # data_dict.pop('len_data_lines')
        return data_dict

    def get_data_info(self, index):
        return self.get_data_info_fn(index)

    def update_branch_id(self, branch_id=0):
        assert isinstance(branch_id, int) and branch_id >= 0
        self.branch_id = branch_id

    def __len__(self):
        return len(self.data_lines)

    def __repr__(self):
        split_str = self.bamboo.__repr__().split('\n')
        bamboo_str = split_str[0]
        for i in range(1, len(split_str)):
            bamboo_str += '\n  ' + split_str[i]

        return 'Dataset(\n  len: {}\n  reader: {}\n  bamboo: {} \n)'.format(len(self), self.reader.__repr__(), bamboo_str)
