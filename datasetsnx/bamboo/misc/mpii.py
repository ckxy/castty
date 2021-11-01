import random
import numpy as np
from ..base_internode import BaseInternode
from utils.bbox_tools import xyxy2xywh, xywh2xyxy


__all__ = ['RandomRescale']


class RandomRescale(BaseInternode):
    def __init__(self, factor=0, distribution='normal', p=1):
        assert 0 < p <= 1
        assert 0 <= factor < 1
        assert distribution in ['uniform', 'normal']

        self.p = p
        self.factor = factor
        self.distribution = distribution

    def __call__(self, data_dict):
        if random.random() < self.p and self.factor > 0:
            if self.distribution == 'uniform':
                factor = 1 - random.uniform(-self.factor, self.factor)
            elif self.distribution == 'normal':
                factor = 1 - np.clip(np.random.randn() * self.factor / 3, -self.factor, self.factor)
        else:
            factor = 1

        box = xyxy2xywh(data_dict['bbox'])
        box[..., 2:4] = data_dict['mpii_scale'] * data_dict['mpii_length'] * factor
        data_dict['bbox'] = xywh2xyxy(box)

        return data_dict

    def reverse(self, **kwargs):
        kwargs['jump'] = 'all'
        return kwargs

    def __repr__(self):
        return 'RandomRescale(factor={}, p={}, distribution={})'.format(self.factor, self.p, self.distribution)

    def rper(self):
        return 'RandomRescale(not available)'
