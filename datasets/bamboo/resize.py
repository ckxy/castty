import cv2
import math
import numpy as np
from PIL import Image
from .base_internode import BaseInternode
from torchvision.transforms.functional import pad
from ..utils.warp_tools import clip_bbox, filter_bbox
from .builder import INTERNODE
from ..utils.warp_tools import get_image_size, is_pil, warp_image
from .warp import WarpResize


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

        nw = int(math.ceil(scale[0] * w))
        nh = int(math.ceil(scale[1] * h))

        if is_pil(data_dict['image']):
            data_dict['image'] = data_dict['image'].resize((nw, nh), Image.BILINEAR)
        else:
            data_dict['image'] = cv2.resize(data_dict['image'], (nw, nh))

        if 'bbox' in data_dict.keys():
            data_dict['bbox'][:, 0] *= scale[0]
            data_dict['bbox'][:, 1] *= scale[1]
            data_dict['bbox'][:, 2] *= scale[0]
            data_dict['bbox'][:, 3] *= scale[1]

        return data_dict

    def __repr__(self):
        return 'Resize(size={}, keep_ratio={}, short={})'.format(self.size, self.keep_ratio, self.short)


@INTERNODE.register_module()
class ResizeAndPadding(WarpResize):
    def __init__(self, size, keep_ratio=True, short=False, warp=False, **kwargs):
        assert len(size) == 2
        assert size[0] > 0 and size[1] > 0

        self.size = size
        self.keep_ratio = keep_ratio
        self.warp = warp
        self.short = short

        if self.warp:
            super(ResizeAndPadding, self).__init__(size, keep_ratio, short, **kwargs)

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

        # print(scale, 'scale')

        tmp = dict()
        if 'bbox' in data_dict.keys():
            tmp['bbox'] = data_dict.pop('bbox')

            tmp['bbox'][:, 0] = tmp['bbox'][:, 0] * scale[0]
            tmp['bbox'][:, 1] = tmp['bbox'][:, 1] * scale[1]
            tmp['bbox'][:, 2] = tmp['bbox'][:, 2] * scale[0]
            tmp['bbox'][:, 3] = tmp['bbox'][:, 3] * scale[1]

        # print(data_dict.keys(), 'aa')

        if self.warp:
            super(ResizeAndPadding, self).__call__(data_dict)
        else:
            nw = int(math.ceil(scale[0] * w))
            nh = int(math.ceil(scale[1] * h))

            if self.short:
                right, bottom = 0, 0
            else:
                l = max(nw, nh)

                right = self.size[0] - nw
                bottom = self.size[1] - nh

            # print(nw, nh, right, bottom, 'nwarp')

            if is_pil(data_dict['image']):
                data_dict['image'] = data_dict['image'].resize((nw, nh), Image.BILINEAR)
                data_dict['image'] = pad(data_dict['image'], (0, 0, right, bottom), 0, 'constant')
            else:
                data_dict['image'] = cv2.resize(data_dict['image'], (nw, nh))
                data_dict['image'] = cv2.copyMakeBorder(data_dict['image'], 0, bottom, 0, right, cv2.BORDER_CONSTANT)

        data_dict.update(tmp)
        # print(data_dict.keys(), 'bb')

        return data_dict

    def reverse(self, **kwargs):
        if 'ori_size' in kwargs.keys():
            h, w = kwargs['ori_size']
            h, w = int(h), int(w)
            scale = self.calc_scale((w, h))

        if 'bbox' in kwargs.keys():
            kwargs['bbox'][:, 0] = kwargs['bbox'][:, 0] / scale[0]
            kwargs['bbox'][:, 1] = kwargs['bbox'][:, 1] / scale[1]
            kwargs['bbox'][:, 2] = kwargs['bbox'][:, 2] / scale[0]
            kwargs['bbox'][:, 3] = kwargs['bbox'][:, 3] / scale[1]

        return kwargs

    def __repr__(self):
        if self.warp:
            s = super(ResizeAndPadding, self).__repr__().replace('WarpResize', 'ResizeAndPadding')
            return s[:-1] + ', warp=True)'
        else:
            return 'ResizeAndPadding(size={}, keep_ratio={}, short={}, warp=False)'.format(self.size, self.keep_ratio, self.short)

    def rper(self):
        return self.__repr__()
