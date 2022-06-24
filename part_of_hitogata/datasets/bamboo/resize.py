import cv2
import math
import random
import numpy as np
from PIL import Image
from .base_internode import BaseInternode
from torchvision.transforms.functional import pad
from ..utils.common import get_image_size, is_pil, clip_bbox, filter_bbox
from .builder import INTERNODE
from ..utils.warp_tools import warp_image
from .warp import WarpResize
from .bamboo import Bamboo
from .builder import build_internode


__all__ = ['Resize', 'Rescale', 'RescaleLimitedByBound', 'ResizeAndPadding']


def resize_image(image, scale):
    w, h = get_image_size(image)
    nw = int(scale[0] * w)
    nh = int(scale[1] * h)

    if is_pil(image):
        image = image.resize((nw, nh), Image.BILINEAR)
    else:
        image = cv2.resize(image, (nw, nh))

    return image


def resize_bbox(bboxes, scale):
    bboxes[:, 0] *= scale[0]
    bboxes[:, 1] *= scale[1]
    bboxes[:, 2] *= scale[0]
    bboxes[:, 3] *= scale[1]

    return bboxes


def resize_poly(polys, scale):
    for i in range(len(polys)):
        polys[i][..., 0] *= scale[0]
        polys[i][..., 1] *= scale[1]

    return polys


@INTERNODE.register_module()
class Resize(BaseInternode):
    def __init__(self, size, keep_ratio=True, short=False, **kwargs):
        assert len(size) == 2
        assert size[0] > 0 and size[1] > 0

        self.size = size
        self.keep_ratio = keep_ratio
        self.short = short

    def calc_scale(self, size):
        w, h = size
        tw, th = self.size
        rw, rh = tw / w, th / h

        if self.keep_ratio:
            if self.short:
                r = max(rh, rw)
                scale = (r, r)
            else:
                r = min(rh, rw)
                scale = (r, r)
        else:
            scale = (rw, rh)
        return scale

    def resize(self, data_dict, scale):
        data_dict['image'] = resize_image(data_dict['image'], scale)

        if 'bbox' in data_dict.keys():
            data_dict['bbox'] = resize_bbox(data_dict['bbox'], scale)
        if 'poly' in data_dict.keys():
            data_dict['poly'] = resize_poly(data_dict['poly'], scale)

        return data_dict

    def __call__(self, data_dict):
        w, h = get_image_size(data_dict['image'])
        scale = self.calc_scale((w, h))
        return self.resize(data_dict, scale)

    def reverse(self, **kwargs):
        if 'resize_and_padding_reverse_flag' not in kwargs.keys():
            return kwargs

        if 'ori_size' in kwargs.keys():
            h, w = kwargs['ori_size']
            h, w = int(h), int(w)
            scale = self.calc_scale((w, h))
            scale = (1 / scale[0], 1 / scale[1])

        if 'bbox' in kwargs.keys():
            kwargs['bbox'] = resize_bbox(kwargs['bbox'], scale)

        if 'poly' in kwargs.keys():
            kwargs['poly'] = resize_poly(kwargs['poly'], scale)

        return kwargs

    def __repr__(self):
        return 'Resize(size={}, keep_ratio={}, short={})'.format(self.size, self.keep_ratio, self.short)


@INTERNODE.register_module()
class Rescale(Resize):
    def __init__(self, ratio_range, mode='range', **kwargs):
        if mode == 'range':
            assert len(ratio_range) == 2 and ratio_range[0] <= ratio_range[1] and ratio_range[0] > 0
        elif mode == 'value':
            assert len(ratio_range) > 1 and min(ratio_range) > 0
        else:
            raise NotImplementedError

        self.ratio_range = ratio_range
        self.mode = mode

    def calc_scale(self, size):
        if self.mode == 'range':
            scale = np.random.random_sample() * (self.ratio_range[1] - self.ratio_range[0]) + self.ratio_range[0]
        elif self.mode == 'value':
            scale = random.choice(self.ratio_range)
        return scale, scale

    def reverse(self, **kwargs):
        return kwargs

    def __repr__(self):
        return 'Rescale(ratio_range={}, mode={})'.format(self.ratio_range, self.mode)


@INTERNODE.register_module()
class RescaleLimitedByBound(Rescale):
    def __init__(self, ratio_range, long_size_bound, short_size_bound, mode='range', **kwargs):
        super(RescaleLimitedByBound, self).__init__(ratio_range, mode, **kwargs)
        assert long_size_bound >= short_size_bound

        self.long_size_bound = long_size_bound
        self.short_size_bound = short_size_bound

    def calc_scale(self, size):
        w, h = size
        scale1 = 1

        if max(h, w) > self.long_size_bound:
            scale1 = self.long_size_bound * 1.0 / max(h, w)

        scale2, _ = super(RescaleLimitedByBound, self).calc_scale(size)
        scale = scale1 * scale2

        if min(h, w) * scale <= self.short_size_bound:
            scale = (self.short_size_bound + 10) * 1.0 / min(h, w)

        return scale, scale

    def reverse(self, **kwargs):
        return kwargs

    def __repr__(self):
        return 'Rescale(ratio_range={}, mode={})'.format(self.ratio_range, self.mode)


@INTERNODE.register_module()
class ResizeAndPadding(Bamboo):
    def __init__(self, resize=None, padding=None, **kwargs):
        assert resize or padding

        self.internodes = []
        if resize:
            assert resize['type'] in ['Resize', 'WarpResize']
            resize['expand'] = True
            self.internodes.append(build_internode(resize))
        else:
            self.internodes.append(build_internode(dict(type='BaseInternode')))

        if padding:
            assert padding['type'] in ['PaddingBySize', 'PaddingByStride']
            self.internodes.append(build_internode(padding))
        else:
            self.internodes.append(build_internode(dict(type='BaseInternode')))

    def reverse(self, **kwargs):
        if 'ori_size' in kwargs.keys():
            h, w = kwargs['ori_size']
            h, w = int(h), int(w)
            ori_size = (h, w)

            if hasattr(self.internodes[0], 'calc_scale'):
                scale = self.internodes[0].calc_scale((w, h))

                nw = math.ceil(scale[0] * w)
                nh = math.ceil(scale[1] * h)

                resize_size = (nh, nw)
            else:
                resize_size = ori_size

        # print(ori_size, resize_size)

        kwargs['resize_and_padding_reverse_flag'] = True

        kwargs['ori_size'] = resize_size
        kwargs = self.internodes[1].reverse(**kwargs)

        kwargs['ori_size'] = ori_size
        kwargs = self.internodes[0].reverse(**kwargs)

        kwargs.pop('resize_and_padding_reverse_flag')

        return kwargs

    def rper(self):
        res = type(self).__name__
        res = res[:1].lower() + res[1:]
        res = res[::-1]
        res = res[:1].upper() + res[1:] + '('

        split_str = [i.__repr__() for i in self.internodes[::-1]]

        for i in range(len(split_str)):
            res += '\n  ' + split_str[i].replace('\n', '\n  ')
        res = '{}\n)'.format(res)
        return res
