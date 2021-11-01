import numpy as np
from utils.bbox_tools import xyxy2xywh
from .base_internode import BaseInternode


__all__ = ['FilterBboxByLength', 'FilterBboxByArea', 'FilterBboxByLengthRatio', 'FilterBboxByAreaRatio', 'FilterBboxByAspectRatio']


class FilterBboxByLength(BaseInternode):
    def __init__(self, min_w, min_h):
        assert min_w >= 0 and min_h >= 0
        assert min_w > 0 or min_h > 0

        self.min_w = min_w
        self.min_h = min_h

    def __call__(self, data_dict):
        if 'bbox' in data_dict.keys():
            xywh = xyxy2xywh(data_dict['bbox'].copy())
            keep = (xywh[..., 2] >= self.min_w) * (xywh[..., 3] >= self.min_h)
            data_dict['bbox'] = data_dict['bbox'][keep]
        return data_dict

    def __repr__(self):
        return 'FilterBboxByLength(min_w={}, min_h={})'.format(self.min_w, self.min_h)

    def rper(self):
        return 'FilterBboxByLength(not available)'


class FilterBboxByArea(BaseInternode):
    def __init__(self, min_a):
        assert min_a > 0
        self.min_a = min_a

    def __call__(self, data_dict):
        if 'bbox' in data_dict.keys():
            xywh = xyxy2xywh(data_dict['bbox'].copy())
            area = xywh[..., 2] * xywh[..., 3]
            keep = area >= self.min_a
            data_dict['bbox'] = data_dict['bbox'][keep]
        return data_dict

    def __repr__(self):
        return 'FilterBboxByArea(min_a={})'.format(self.min_a)

    def rper(self):
        return 'FilterBboxByArea(not available)'


class FilterBboxByLengthRatio(BaseInternode):
    def __init__(self, min_w=None, min_h=None):
        assert 0 <= min_w < 1 and 0 <= min_h < 1
        assert min_w > 0 or min_h > 0

        self.min_w = min_w
        self.min_h = min_h

    def __call__(self, data_dict):
        if 'bbox' in data_dict.keys():
            xywh = xyxy2xywh(data_dict['bbox'].copy())
            w, h = data_dict['image'].size
            # print(self.min_w * w, self.min_h * h)
            keep = (xywh[..., 2] >= (self.min_w * w)) * (xywh[..., 3] >= (self.min_h * h))
            data_dict['bbox'] = data_dict['bbox'][keep]
        return data_dict

    def __repr__(self):
        return 'FilterBboxByLengthRatio(min_w={}, min_h={})'.format(self.min_w, self.min_h)

    def rper(self):
        return 'FilterBboxByLengthRatio(not available)'


class FilterBboxByAreaRatio(BaseInternode):
    def __init__(self, min_a):
        assert min_a > 0
        self.min_a = min_a

    def __call__(self, data_dict):
        if 'bbox' in data_dict.keys():
            xywh = xyxy2xywh(data_dict['bbox'].copy())
            area = xywh[..., 2] * xywh[..., 3]
            w, h = data_dict['image'].size
            keep = area >= (self.min_a * w * h)
            data_dict['bbox'] = data_dict['bbox'][keep]
        return data_dict

    def __repr__(self):
        return 'FilterBboxByAreaRatio(min_a={})'.format(self.min_a)

    def rper(self):
        return 'FilterBboxByAreaRatio(not available)'


class FilterBboxByAspectRatio(BaseInternode):
    def __init__(self, aspect_ratio):
        assert isinstance(aspect_ratio, tuple) and len(aspect_ratio) == 2
        assert aspect_ratio[0] > 0 and aspect_ratio[1] > 0 and aspect_ratio[0] <= aspect_ratio[1]
        self.aspect_ratio = aspect_ratio

    def __call__(self, data_dict):
        if 'bbox' in data_dict.keys():
            xywh = xyxy2xywh(data_dict['bbox'].copy())
            keep = (xywh[..., 2] <= (xywh[..., 3] * self.aspect_ratio[1])) * (xywh[..., 2] >= (xywh[..., 3] * self.aspect_ratio[0]))
            data_dict['bbox'] = data_dict['bbox'][keep]
        return data_dict

    def __repr__(self):
        return 'FilterBboxByAspectRatio(aspect_ratio={})'.format(self.aspect_ratio)

    def rper(self):
        return 'FilterBboxByAspectRatio(not available)'
