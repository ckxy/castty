import cv2
import math
import numbers
import numpy as np
from .base_internode import BaseInternode
from torchvision.transforms.functional import pad
from ..utils.warp_tools import clip_bbox, filter_bbox, is_pil, get_image_size
from .builder import INTERNODE


__all__ = ['Padding', 'PaddingByStride']


@INTERNODE.register_module()
class Padding(BaseInternode):
    def __init__(self, padding, fill=0, padding_mode='constant', **kwargs):
        assert isinstance(padding, tuple) and len(padding) == 4
        assert isinstance(fill, (numbers.Number, tuple))

        self.padding_modes = dict(
            constant=cv2.BORDER_CONSTANT, 
            edge=cv2.BORDER_REPLICATE, 
            reflect=cv2.BORDER_REFLECT_101, 
            symmetric=cv2.BORDER_REFLECT
        )
        assert padding_mode in self.padding_modes.keys()

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, data_dict):
        if is_pil(data_dict['image']):
            data_dict['image'] = pad(data_dict['image'], self.padding, self.fill, self.padding_mode)
        else:
            left, top, right, bottom = self.padding
            if data_dict['image'].ndim == 2:
                assert isinstance(fill, numbers.Number)
            else:
                assert len(self.fill) == data_dict['image'].shape[-1]
            data_dict['image'] = cv2.copyMakeBorder(data_dict['image'], top, bottom, left, right, self.padding_modes[self.padding_mode], value=self.fill)

        if 'bbox' in data_dict.keys():
            data_dict['bbox'][:, 0] += self.padding[0]
            data_dict['bbox'][:, 1] += self.padding[1]
            data_dict['bbox'][:, 2] += self.padding[0]
            data_dict['bbox'][:, 3] += self.padding[1]
        return data_dict

    def __repr__(self):
        return 'Padding(padding={}, fill={}, padding_mode={})'.format(self.padding, self.fill, self.padding_mode)


@INTERNODE.register_module()
class PaddingBySize(BaseInternode):
    def __init__(self, size, fill=0, padding_mode='constant', center=False, **kwargs):
        assert isinstance(size, tuple) and len(size) == 2
        assert isinstance(fill, (numbers.Number, tuple))

        self.padding_modes = dict(
            constant=cv2.BORDER_CONSTANT, 
            edge=cv2.BORDER_REPLICATE, 
            reflect=cv2.BORDER_REFLECT_101, 
            symmetric=cv2.BORDER_REFLECT
        )
        assert padding_mode in self.padding_modes.keys()

        self.size = size
        self.fill = fill
        self.padding_mode = padding_mode
        self.center = center

    def calc_padding(self, w, h):
        if self.center:
            left = max((self.size[0] - w) // 2, 0)
            right = max(self.size[0] - w, 0) - left
            top = max((self.size[1] - h) // 2, 0)
            bottom = max(self.size[1] - h, 0) - top
        else:
            left = 0
            right = max(self.size[0] - w, 0)
            top = 0
            bottom = max(self.size[1] - h, 0)
        return left, right, top, bottom

    def __call__(self, data_dict):
        w, h = get_image_size(data_dict['image'])
        # print(w, h, self.size)

        left, right, top, bottom = self.calc_padding(w, h)
        # print(left, right, top, bottom)
        # exit()

        if is_pil(data_dict['image']):
            data_dict['image'] = pad(data_dict['image'], (left, top, right, bottom), self.fill, self.padding_mode)
        else:
            if data_dict['image'].ndim == 2:
                assert isinstance(fill, numbers.Number)
            else:
                assert len(self.fill) == data_dict['image'].shape[-1]
            data_dict['image'] = cv2.copyMakeBorder(data_dict['image'], top, bottom, left, right, self.padding_modes[self.padding_mode], value=self.fill)

        if 'bbox' in data_dict.keys():
            data_dict['bbox'][:, 0] += left
            data_dict['bbox'][:, 1] += top
            data_dict['bbox'][:, 2] += left
            data_dict['bbox'][:, 3] += top
        return data_dict

    def reverse(self, **kwargs):
        if 'resize_and_padding_reverse_flag' not in kwargs.keys():
            return kwargs

        if 'ori_size' in kwargs.keys():
            h, w = kwargs['ori_size']
            h, w = int(h), int(w)

            left, _, top, _ = self.calc_padding(w, h)

        if 'bbox' in kwargs.keys():
            kwargs['bbox'][:, 0] -= left
            kwargs['bbox'][:, 1] -= top
            kwargs['bbox'][:, 2] -= left
            kwargs['bbox'][:, 3] -= top

            boxes = clip_bbox(kwargs['bbox'], (w, h))
            keep = filter_bbox(boxes)
            kwargs['bbox'] = boxes[keep]

            if 'bbox_meta' in kwargs.keys():
                kwargs['bbox_meta'].filter(keep)

        return kwargs

    def __repr__(self):
        return 'PaddingBySize(size={}, fill={}, padding_mode={}, center={})'.format(self.size, self.fill, self.padding_mode, self.center)


@INTERNODE.register_module()
class PaddingByStride(BaseInternode):
    def __init__(self, stride, fill=0, padding_mode='constant', center=False, **kwargs):
        assert isinstance(stride, int) and stride > 0
        assert isinstance(fill, (numbers.Number, tuple))

        self.padding_modes = dict(
            constant=cv2.BORDER_CONSTANT, 
            edge=cv2.BORDER_REPLICATE, 
            reflect=cv2.BORDER_REFLECT_101, 
            symmetric=cv2.BORDER_REFLECT
        )
        assert padding_mode in self.padding_modes.keys()

        self.stride = stride
        self.fill = fill
        self.padding_mode = padding_mode
        self.center = center

    def calc_padding(self, w, h):
        nw = math.ceil(w / self.stride) * self.stride
        nh = math.ceil(h / self.stride) * self.stride

        if self.center:
            left = (nw - w) // 2
            right = nw - w - left
            top = (nh - h) // 2
            bottom = nh - h - top
        else:
            left = 0
            right = nw - w
            top = 0
            bottom = nh - h
        return left, right, top, bottom

    def __call__(self, data_dict):
        w, h = get_image_size(data_dict['image'])

        left, right, top, bottom = self.calc_padding(w, h)
        # print(left, right, top, bottom)

        if is_pil(data_dict['image']):
            data_dict['image'] = pad(data_dict['image'], (left, top, right, bottom), self.fill, self.padding_mode)
        else:
            if data_dict['image'].ndim == 2:
                assert isinstance(fill, numbers.Number)
            else:
                assert len(self.fill) == data_dict['image'].shape[-1]
            data_dict['image'] = cv2.copyMakeBorder(data_dict['image'], top, bottom, left, right, self.padding_modes[self.padding_mode], value=self.fill)

        if 'bbox' in data_dict.keys():
            data_dict['bbox'][:, 0] += left
            data_dict['bbox'][:, 1] += top
            data_dict['bbox'][:, 2] += left
            data_dict['bbox'][:, 3] += top

        return data_dict

    def reverse(self, **kwargs):
        if 'resize_and_padding_reverse_flag' not in kwargs.keys():
            return kwargs

        if 'ori_size' in kwargs.keys():
            h, w = kwargs['ori_size']
            h, w = int(h), int(w)

            left, _, top, _ = self.calc_padding(w, h)

        if 'bbox' in kwargs.keys():
            kwargs['bbox'][:, 0] -= left
            kwargs['bbox'][:, 1] -= top
            kwargs['bbox'][:, 2] -= left
            kwargs['bbox'][:, 3] -= top

            boxes = clip_bbox(kwargs['bbox'], (w, h))
            keep = filter_bbox(boxes)
            kwargs['bbox'] = boxes[keep]

            if 'bbox_meta' in kwargs.keys():
                kwargs['bbox_meta'].filter(keep)

        return kwargs

    def __repr__(self):
        return 'PaddingByStride(stride={}, fill={}, padding_mode={}, center={})'.format(self.stride, self.fill, self.padding_mode, self.center)
