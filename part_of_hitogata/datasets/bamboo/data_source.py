import copy
from .builder import INTERNODE
from .base_internode import BaseInternode


__all__ = ['DataSource']


@INTERNODE.register_module()
class DataSource(BaseInternode):
    def forward(self, data_dict):
        if 'reader' in data_dict.keys():
            index = data_dict.pop('index')
            reader = data_dict.pop('reader')
            data_dict.pop('len_data_lines')

            tmp = copy.deepcopy(data_dict)
            tmp.update(reader[index])
            return tmp
        return data_dict

    def backward(self, data_dict):
        return data_dict
