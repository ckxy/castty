import torch
import numpy as np
from .base_internode import BaseInternode
from .builder import INTERNODE


__all__ = ['OneHotEncode']


@INTERNODE.register_module()
class OneHotEncode(BaseInternode):
    def __init__(self, num_classes):
        self.num_classes = kwargs['num_classes']

    def __call__(self, data_dict):
        data_dict['label'] = np.eye(self.num_classes)[data_dict['label']].astype(np.float32)
        return data_dict

    def __repr__(self):
        return 'OneHotEncode(num_classes={})'.format(self.num_classes)

    def rper(self):
        return 'OneHotEncode(not available)'
