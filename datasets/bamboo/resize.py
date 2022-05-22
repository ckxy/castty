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


# @INTERNODE.register_module()
# class ResizeAndPadding(WarpResize):
#     def __init__(self, size, keep_ratio=True, short=False, warp=False, **kwargs):
#         assert len(size) == 2
#         assert size[0] > 0 and size[1] > 0

#         self.size = size
#         self.keep_ratio = keep_ratio
#         self.warp = warp
#         self.short = short

#         if self.warp:
#             super(ResizeAndPadding, self).__init__(size, keep_ratio, short, **kwargs)

#     def calc_scale(self, size):
#         w, h = size
#         tw, th = self.size
#         rw, rh = tw / w, th / h

#         if self.keep_ratio:
#             if self.short:
#                 r = max(rh, rw)
#                 scale = (r, r)
#             else:
#                 r = min(rh, rw)
#                 scale = (r, r)
#         else:
#             scale = (rw, rh)
#         return scale

#     def __call__(self, data_dict):
#         w, h = get_image_size(data_dict['image'])
#         scale = self.calc_scale((w, h))

#         # print(scale, 'scale')

#         tmp = dict()
#         if 'bbox' in data_dict.keys():
#             tmp['bbox'] = data_dict.pop('bbox')

#             tmp['bbox'][:, 0] = tmp['bbox'][:, 0] * scale[0]
#             tmp['bbox'][:, 1] = tmp['bbox'][:, 1] * scale[1]
#             tmp['bbox'][:, 2] = tmp['bbox'][:, 2] * scale[0]
#             tmp['bbox'][:, 3] = tmp['bbox'][:, 3] * scale[1]

#         # print(data_dict.keys(), 'aa')

#         if self.warp:
#             super(ResizeAndPadding, self).__call__(data_dict)
#         else:
#             nw = int(math.ceil(scale[0] * w))
#             nh = int(math.ceil(scale[1] * h))

#             if self.short:
#                 right, bottom = 0, 0
#             else:
#                 l = max(nw, nh)

#                 right = self.size[0] - nw
#                 bottom = self.size[1] - nh

#             # print(nw, nh, right, bottom, 'nwarp')

#             if is_pil(data_dict['image']):
#                 data_dict['image'] = data_dict['image'].resize((nw, nh), Image.BILINEAR)
#                 data_dict['image'] = pad(data_dict['image'], (0, 0, right, bottom), 0, 'constant')
#             else:
#                 data_dict['image'] = cv2.resize(data_dict['image'], (nw, nh))
#                 data_dict['image'] = cv2.copyMakeBorder(data_dict['image'], 0, bottom, 0, right, cv2.BORDER_CONSTANT)

#         data_dict.update(tmp)
#         # print(data_dict.keys(), 'bb')

#         return data_dict

#     def reverse(self, **kwargs):
#         if 'ori_size' in kwargs.keys():
#             h, w = kwargs['ori_size']
#             h, w = int(h), int(w)
#             scale = self.calc_scale((w, h))

#         if 'bbox' in kwargs.keys():
#             kwargs['bbox'][:, 0] = kwargs['bbox'][:, 0] / scale[0]
#             kwargs['bbox'][:, 1] = kwargs['bbox'][:, 1] / scale[1]
#             kwargs['bbox'][:, 2] = kwargs['bbox'][:, 2] / scale[0]
#             kwargs['bbox'][:, 3] = kwargs['bbox'][:, 3] / scale[1]

#         return kwargs

#     def __repr__(self):
#         if self.warp:
#             s = super(ResizeAndPadding, self).__repr__().replace('WarpResize', 'ResizeAndPadding')
#             return s[:-1] + ', warp=True)'
#         else:
#             return 'ResizeAndPadding(size={}, keep_ratio={}, short={}, warp=False)'.format(self.size, self.keep_ratio, self.short)

#     def rper(self):
#         return self.__repr__()
