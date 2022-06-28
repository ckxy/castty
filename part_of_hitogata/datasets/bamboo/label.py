import torch
import numpy as np
from .base_internode import BaseInternode
from .builder import INTERNODE


__all__ = ['OneHotEncode', 'OneHotDecode']


@INTERNODE.register_module()
class OneHotEncode(BaseInternode):
    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes

    def __call__(self, data_dict):
        data_dict['label'] = np.eye(self.num_classes)[data_dict['label']].astype(np.float32)
        return data_dict

    def reverse(self, **kwargs):
        if 'label' in kwargs.keys():
            kwargs['label'] = int(np.argmax(kwargs['label']))
        return kwargs

    def __repr__(self):
        return 'OneHotEncode(num_classes={})'.format(self.num_classes)

    def rper(self):
        return 'OneHotDecode()'

@INTERNODE.register_module()
class OneHotDecode(BaseInternode):
    def __call__(self, data_dict):
        if len(data_dict['label']) > 1:
            data_dict = self.reverse(data_dict)
        return data_dict

    def reverse(self, **kwargs):
        if 'label' in kwargs.keys():
            kwargs['label'] = int(np.argmax(kwargs['label']))
        return kwargs

    def __repr__(self):
        return 'OneHotDecode()'

    def rper(self):
        return 'OneHotDecode()'
