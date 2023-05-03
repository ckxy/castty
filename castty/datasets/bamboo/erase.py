import cv2
import math
import random
import numpy as np
from .builder import INTERNODE
from .mixin import DataAugMixin
from .base_internode import BaseInternode
from PIL import Image, ImageOps, ImageDraw
from ..utils.common import get_image_size, is_pil


__all__ = ['RandomErasing', 'GridMask']


class ErasingInternode(DataAugMixin, BaseInternode):
    def __init__(self, tag_mapping=dict(image=['image'], mask=['mask']), **kwargs):
        forward_mapping = dict(
            image=self.forward_image,
            mask=self.forward_mask,
        )
        backward_mapping = dict()
        super(ErasingInternode, self).__init__(tag_mapping, forward_mapping, backward_mapping, **kwargs)

    def forward_image(self, image, meta, intl_erase_mask, intl_erase_bgd, **kwargs):
        if intl_erase_mask is None:
            return image, meta

        if is_pil(image):
            image = Image.composite(image, intl_erase_bgd, intl_erase_mask)
        else:
            image = Image.fromarray(image)
            image = Image.composite(image, intl_erase_bgd, intl_erase_mask)
            image = np.array(image)

        return image, meta

    def forward_mask(self, mask, meta, intl_erase_mask, intl_erase_bgd, **kwargs):
        if intl_erase_mask is None:
            return mask, meta

        intl_erase_mask = (np.asarray(intl_erase_mask) > 0).astype(np.int32)
        w, h = get_image_size(mask)
        bgd = np.zeros((h, w), np.int32)
        mask = mask * intl_erase_mask + bgd * (1 - intl_erase_mask)

        return mask, meta


@INTERNODE.register_module()
class RandomErasing(ErasingInternode):
    def __init__(self, scale=(0.02, 0.33), ratio=(0.3, 3.3), offset=False, value=(0, 0, 0), **kwargs):
        assert isinstance(value, tuple)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")

        self.scale = scale
        self.ratio = ratio
        self.offset = offset
        self.value = value

        super(RandomErasing, self).__init__(**kwargs)

    def calc_intl_param_forward(self, data_dict):
        assert 'point' not in data_dict.keys() and 'bbox' not in data_dict.keys()

        param = dict(intl_erase_mask=None, intl_erase_bgd=None)

        w, h = get_image_size(data_dict['image'])
        area = w * h
        for attempt in range(10):
            erase_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

            new_h = int(round(math.sqrt(erase_area * aspect_ratio)))
            new_w = int(round(math.sqrt(erase_area / aspect_ratio)))

            if new_h < h and new_w < w:
                y = random.randint(0, h - new_h)
                x = random.randint(0, w - new_w)

                param['intl_erase_mask'] = Image.new("L", get_image_size(data_dict['image']), 255)
                draw = ImageDraw.Draw(param['intl_erase_mask'])
                draw.rectangle((x, y, x + new_w, y + new_h), fill=0)

                if 'image' in data_dict.keys():
                    if self.offset:
                        offset = 2 * (np.random.rand(h, w) - 0.5)
                        offset = np.uint8(offset * 255)
                        param['intl_erase_bgd'] = Image.fromarray(offset).convert('RGB')
                    else:
                        param['intl_erase_bgd'] = Image.new('RGB', get_image_size(data_dict['image']), self.value)
                break

        return param

    def __repr__(self):
        if self.offset:
            return 'RandomErasing(scale={}, ratio={}, offset={})'.format(self.scale, self.ratio, self.offset)
        else:
            return 'RandomErasing(scale={}, ratio={}, value={})'.format(self.scale, self.ratio, self.value)


@INTERNODE.register_module()
class GridMask(ErasingInternode):
    def __init__(self, use_w=True, use_h=True, rotate=0, offset=False, invert=False, ratio=1, **kwargs):
        assert 0 <= rotate < 90

        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.invert = invert
        self.ratio = ratio

        super(GridMask, self).__init__(**kwargs)

    def calc_intl_param_forward(self, data_dict):
        assert 'point' not in data_dict.keys()

        w, h = get_image_size(data_dict['image'])

        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, min(h, w))

        if self.ratio == 1:
            l = np.random.randint(1, d)
        else:
            l = min(max(int(d * self.ratio + 0.5), 1), d - 1)

        mask = np.ones((hh, ww), np.float32)

        st_h = np.random.randint(d)
        st_w = np.random.randint(d)

        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + l, hh)
                mask[s:t, :] = 0

        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + l, ww)
                mask[:, s:t] = 0

        mask = Image.fromarray(np.uint8(mask * 255))
        if not self.invert:
            mask = ImageOps.invert(mask)

        if self.rotate != 0:
            r = np.random.randint(self.rotate)
            mask = mask.rotate(r)

        param = dict()
        param['intl_erase_mask'] = mask.crop(((ww - w) // 2, (hh - h) // 2, (ww - w) // 2 + w, (hh - h) // 2 + h))

        if 'image' in data_dict.keys():
            if self.offset:
                offset = 2 * (np.random.rand(h, w) - 0.5)
                offset = np.uint8(offset * 255)
                param['intl_erase_bgd'] = Image.fromarray(offset).convert('RGB')
            else:
                param['intl_erase_bgd'] = Image.new('RGB', get_image_size(data_dict['image']), 0)

        return param

    def __repr__(self):
        return 'GridMask(use_h={}, use_w={}, ratio={}, rotate={}, offset={}, invert={})'.format(self.use_h, self.use_w, self.ratio, self.rotate, self.offset, self.invert)
