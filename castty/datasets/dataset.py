import torch.utils.data as data
from .bamboo.builder import build_bamboo
from .readers.builder import build_reader


class Dataset(data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg

        self.reader = build_reader(cfg.reader)
        self._info = self.reader.info
        # self.bamboo = Bamboo(cfg.internodes, tag_mapping=self._info['tag_mapping'])
        # self.bamboo = build_internode(dict(type='Bamboo', internodes=cfg.internodes), tag_mapping=self._info['tag_mapping'])
        tag_mapping = cfg.tag_mapping if cfg.tag_mapping else self._info['tag_mapping']
        self.bamboo = build_bamboo(internodes=cfg.internodes, tag_mapping=tag_mapping)

        forcat = self._info.pop('forcat')
        self._info.update(forcat)

        self.branch_id = 0

    @property
    def info(self):
        return self._info

    def __getitem__(self, index):
        data_dict = dict(reader=self.reader, index=index, len_data_lines=len(self), intl_branch_id=self.branch_id)
        data_dict = self.bamboo(data_dict)

        data_dict.pop('intl_branch_id')
        if 'intl_group_id' in data_dict.keys():
            data_dict.pop('intl_group_id')
        
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
