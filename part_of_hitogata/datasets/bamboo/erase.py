import cv2
import math
import random
import numpy as np
from .builder import INTERNODE
from .base_internode import BaseInternode
from PIL import Image, ImageOps, ImageDraw
from ..utils.common import get_image_size, is_pil


__all__ = ['RandomErasing', 'GridMask']


@INTERNODE.register_module()
class RandomErasing(BaseInternode):
    def __init__(self, scale=(0.02, 0.33), ratio=(0.3, 3.3), offset=False, value=(0, 0, 0)):
        assert isinstance(value, tuple)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")

        self.scale = scale
        self.ratio = ratio
        self.offset = offset
        self.value = value

    def get_params(self, w, h):
        area = w * h
        for attempt in range(10):
            erase_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

            new_h = int(round(math.sqrt(erase_area * aspect_ratio)))
            new_w = int(round(math.sqrt(erase_area / aspect_ratio)))

            if new_h < h and new_w < w:
                y = random.randint(0, h - new_h)
                x = random.randint(0, w - new_w)
                return x, y, new_w, new_h

        return None, None, None, None

    def __call__(self, data_dict):
        assert 'point' not in data_dict.keys() and 'bbox' not in data_dict.keys()

        w, h = get_image_size(data_dict['image'])
        x, y, new_w, new_h = self.get_params(w, h)

        if x is None:
            return data_dict

        if is_pil(data_dict['image']):
            mask = Image.new("L", data_dict['image'].size, 255)
            draw = ImageDraw.Draw(mask)
            draw.rectangle((x, y, x + new_w, y + new_h), fill=0)

            if self.offset:
                offset = 2 * (np.random.rand(h, w) - 0.5)
                offset = np.uint8(offset * 255)
                bgd = Image.fromarray(offset).convert(data_dict['image'].mode)
            else:
                bgd = Image.new(data_dict['image'].mode, data_dict['image'].size, self.value)

            data_dict['image'] = Image.composite(data_dict['image'], bgd, mask)
        else:
            mask = np.ones((h, w, 3), np.uint8)
            mask = cv2.rectangle(mask, (x, y), (x + new_w, y + new_h), (0, 0, 0), -1)

            if self.offset:
                offset = 2 * (np.random.rand(h, w) - 0.5)
                offset = np.uint8(offset * 255)
                bgd = offset[..., np.newaxis]
                bgd = np.repeat(bgd, 3, axis=-1)
            else:
                bgd = np.ones((h, w, 3), np.uint8)
                bgd[..., 0] = self.value[0]
                bgd[..., 1] = self.value[1]
                bgd[..., 2] = self.value[2]

            data_dict['image'] = data_dict['image'] * mask + bgd * (1 - mask)

        if 'mask' in data_dict.keys():
            if is_pil(data_dict['image']):
                mask = (np.asarray(mask) > 0).astype(np.int32)
            else:
                mask = mask[..., 0]
            bgd = np.zeros((h, w), np.int32)
            data_dict['mask'] = data_dict['mask'] * mask + bgd * (1 - mask)

        return data_dict

    def __repr__(self):
        if self.offset:
            return 'RandomErasing(scale={}, ratio={}, offset={})'.format(self.scale, self.ratio, self.offset)
        else:
            return 'RandomErasing(scale={}, ratio={}, value={})'.format(self.scale, self.ratio, self.value)


@INTERNODE.register_module()
class GridMask(BaseInternode):
    def __init__(self, use_w=True, use_h=True, rotate=0, offset=False, invert=False, ratio=1):
        assert 0 <= rotate < 90

        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.invert = invert
        self.ratio = ratio

    def __call__(self, data_dict):
        assert 'point' not in data_dict.keys()

        w, h = get_image_size(data_dict['image'])

        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, min(h, w))

        if self.ratio == 1:
            l = np.random.randint(1, d)
        else:
            l = min(max(int(d * self.ratio + 0.5), 1), d - 1)

        # ch = len(data_dict['image'].getbands())
        # print(l, d)

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

        if self.offset:
            offset = 2 * (np.random.rand(h, w) - 0.5)
            offset = np.uint8(offset * 255)
            if is_pil(data_dict['image']):
                bgd = Image.fromarray(offset).convert(data_dict['image'].mode)
            else:
                bgd = offset[..., np.newaxis]
                bgd = np.repeat(bgd, 3, axis=-1)
        else:
            if is_pil(data_dict['image']):
                bgd = Image.new(data_dict['image'].mode, data_dict['image'].size, 0)
            else:
                bgd = np.zeros(data_dict['image'].shape).astype(np.uint8)

        mask = Image.fromarray(np.uint8(mask * 255))
        if not self.invert:
            mask = ImageOps.invert(mask)

        if self.rotate != 0:
            r = np.random.randint(self.rotate)
            mask = mask.rotate(r)

        mask = mask.crop(((ww - w) // 2, (hh - h) // 2, (ww - w) // 2 + w, (hh - h) // 2 + h))

        if is_pil(data_dict['image']):
            data_dict['image'] = Image.composite(data_dict['image'], bgd, mask)
        else:
            mask_cv2 = mask.convert('RGB')
            mask_cv2 = (np.asarray(mask_cv2) > 0).astype(np.uint8)
            data_dict['image'] = data_dict['image'] * mask_cv2 + bgd * (1 - mask_cv2)

        if 'mask' in data_dict.keys():
            mask = (np.asarray(mask) > 0).astype(np.int32)
            bgd = np.zeros((h, w), np.int32)
            data_dict['mask'] = data_dict['mask'] * mask + bgd * (1 - mask)

        return data_dict

    def __repr__(self):
        return 'GridMask(use_h={}, use_w={}, ratio={}, rotate={}, offset={}, invert={})'.format(self.use_h, self.use_w, self.ratio, self.rotate, self.offset, self.invert)

