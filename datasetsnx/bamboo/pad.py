import math
import numbers
import numpy as np
from .base_internode import BaseInternode
from torchvision.transforms.functional import pad
from .utils.warp_tools import clip_bbox, filter_bbox


__all__ = ['Padding', 'PaddingByStride']


class Padding(BaseInternode):
    def __init__(self, padding, fill=0, padding_mode='constant'):
        assert isinstance(padding, tuple) and len(padding) == 4
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, data_dict):
        # if ('point' in data_dict.keys() or 'bbox' in data_dict.keys()) and self.padding_mode in ['reflect', 'symmetric']:
        #     raise ValueError

        data_dict['image'] = pad(data_dict['image'], self.padding, self.fill, self.padding_mode)

        if 'mask' in data_dict.keys():
            # padding_mode = self.padding_mode if self.padding_mode != 'edge' else 'constant'
            data_dict['mask'] = pad(data_dict['mask'], self.padding, 0, 'constant')

        if 'bbox' in data_dict.keys():
            data_dict['bbox'][:, 0] += self.padding[0]
            data_dict['bbox'][:, 1] += self.padding[1]
            data_dict['bbox'][:, 2] += self.padding[0]
            data_dict['bbox'][:, 3] += self.padding[1]

        if 'point' in data_dict.keys():
            data_dict['point'][:, 0] += self.padding[0]
            data_dict['point'][:, 1] += self.padding[1]

        # if 'polygon' in data_dict.keys():
        #     data_dict['polygon'][..., 0] += t_w
        #     data_dict['polygon'][..., 1] += t_h

        return data_dict

    def __repr__(self):
        return 'Padding(padding={}, fill={}, padding_mode={})'.format(self.padding, self.fill, self.padding_mode)

    def rper(self):
        return 'Padding(not available)'


class PaddingByStride(BaseInternode):
    def __init__(self, stride):
        assert isinstance(stride, int) and stride > 0
        self.stride = stride

    def __call__(self, data_dict):
        w, h = data_dict['image'].size
        nw = math.ceil(w / self.stride) * self.stride
        nh = math.ceil(h / self.stride) * self.stride

        right = nw - w
        bottom = nh - h

        data_dict['image'] = pad(data_dict['image'], (0, 0, right, bottom), 0, 'constant')

        if 'mask' in data_dict.keys():
            data_dict['mask'] = pad(data_dict['mask'], (0, 0, right, bottom), 0, 'constant')

        return data_dict

    def reverse(self, **kwargs):
        if 'training' in kwargs.keys() and kwargs['training']:
            return kwargs

        h, w = kwargs['ori_size']
        h, w = int(h), int(w)
        nw = math.ceil(w / self.stride) * self.stride
        nh = math.ceil(h / self.stride) * self.stride

        if 'image' in kwargs.keys():
            kwargs['image'] = kwargs['image'].crop((0, 0, w, h))

        if 'mask' in kwargs.keys():
            kwargs['mask'] = kwargs['mask'].crop((0, 0, w, h))

        if 'bbox' in kwargs.keys():
            boxes = kwargs['bbox'][:, :4]
            other = kwargs['bbox'][:, 4:]
            boxes = clip_bbox(boxes, (w, h))
            keep = filter_bbox(boxes)

            kwargs['bbox'] = np.concatenate((boxes, other), axis=-1)[keep]

        if 'point' in kwargs.keys():
            pass

        return kwargs

    def __repr__(self):
        return 'PaddingByStride(stride={}))'.format(self.stride)

    def rper(self):
        return self.__repr__()
