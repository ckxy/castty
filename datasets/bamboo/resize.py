import cv2
import math
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


__all__ = ['Resize', 'ResizeAndPadding']


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

    def __call__(self, data_dict):
        w, h = get_image_size(data_dict['image'])
        scale = self.calc_scale((w, h))

        nw = math.ceil(scale[0] * w)
        nh = math.ceil(scale[1] * h)

        if is_pil(data_dict['image']):
            data_dict['image'] = data_dict['image'].resize((nw, nh), Image.BILINEAR)
        else:
            data_dict['image'] = cv2.resize(data_dict['image'], (nw, nh))

        if 'bbox' in data_dict.keys():
            data_dict['bbox'][:, 0] *= scale[0]
            data_dict['bbox'][:, 1] *= scale[1]
            data_dict['bbox'][:, 2] *= scale[0]
            data_dict['bbox'][:, 3] *= scale[1]

        if 'poly' in data_dict.keys():
            for i in range(len(data_dict['poly'])):
                data_dict['poly'][i][..., 0] *= scale[0]
                data_dict['poly'][i][..., 1] *= scale[1]
            # print(data_dict['poly'])
            # exit()

        return data_dict

    def reverse(self, **kwargs):
        if 'resize_and_padding_reverse_flag' not in kwargs.keys():
            return kwargs

        if 'ori_size' in kwargs.keys():
            h, w = kwargs['ori_size']
            h, w = int(h), int(w)
            scale = self.calc_scale((w, h))

        if 'bbox' in kwargs.keys():
            kwargs['bbox'][:, 0] = kwargs['bbox'][:, 0] / scale[0]
            kwargs['bbox'][:, 1] = kwargs['bbox'][:, 1] / scale[1]
            kwargs['bbox'][:, 2] = kwargs['bbox'][:, 2] / scale[0]
            kwargs['bbox'][:, 3] = kwargs['bbox'][:, 3] / scale[1]

        if 'poly' in kwargs.keys():
            for i in range(len(kwargs['poly'])):
                kwargs['poly'][i][..., 0] = kwargs['poly'][i][..., 0] / scale[0]
                kwargs['poly'][i][..., 1] = kwargs['poly'][i][..., 1] / scale[1]

        return kwargs

    def __repr__(self):
        return 'Resize(size={}, keep_ratio={}, short={})'.format(self.size, self.keep_ratio, self.short)


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
