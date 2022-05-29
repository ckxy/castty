from .base_internode import BaseInternode
from .builder import INTERNODE


__all__ = ['DataSource']


@INTERNODE.register_module()
class DataSource(BaseInternode):
    def __call__(self, data_dict):
        if 'reader' in data_dict.keys():
            index = data_dict.pop('index')
            reader = data_dict.pop('reader')
            data_dict.pop('len_data_lines')
            data_dict = reader(index)
        return data_dict
