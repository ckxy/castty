import math
import random
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from .base_internode import BaseInternode


__all__ = ['RandomErasing', 'GridMask']


class RandomErasing(BaseInternode):
    def __init__(self, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(0, 0, 0), p=1):
        assert isinstance(value, tuple)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")

        self.p = p
        self.scale = scale
        self.ratio = ratio
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

        if random.random() < self.p:
            w, h = data_dict['image'].size
            x, y, new_w, new_h = self.get_params(w, h)

            if x is None:
                return data_dict

            mask = Image.new("L", data_dict['image'].size, 255)
            draw = ImageDraw.Draw(mask)
            draw.rectangle((x, y, x + new_w, y + new_h), fill=0)
            bgd = Image.new(data_dict['image'].mode, data_dict['image'].size, self.value)

            data_dict['image'] = Image.composite(data_dict['image'], bgd, mask)

            if 'mask' in data_dict.keys():
                mask_bgd = Image.new(data_dict['mask'].mode, data_dict['mask'].size, 0)
                data_dict['mask'] = Image.composite(data_dict['mask'], mask_bgd, mask)

        return data_dict

    def __repr__(self):
        return 'RandomErasing(p={}, scale={}, ratio={}, value={})'.format(self.p, self.scale, self.ratio, self.value)

    def rper(self):
        return 'RandomErasing(not available)'


class GridMask(BaseInternode):
    def __init__(self, use_w=True, use_h=True, rotate=0, offset=False, invert=False, ratio=1, p=1):
        assert 0 < p <= 1
        assert 0 <= rotate < 360

        self.p = p
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.invert = invert
        self.ratio = ratio

    def __call__(self, data_dict):
        assert 'point' not in data_dict.keys()

        if random.random() < self.p:
            w, h = data_dict['image'].size

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
                bgd = Image.fromarray(np.uint8(offset * 255)).convert(data_dict['image'].mode)
            else:
                bgd = Image.new(data_dict['image'].mode, data_dict['image'].size, 0)

            mask = Image.fromarray(np.uint8(mask * 255))
            if not self.invert:
                mask = ImageOps.invert(mask)

            if self.rotate != 0:
                r = np.random.randint(self.rotate)
                mask = mask.rotate(r)

            mask = mask.crop(((ww - w) // 2, (hh - h) // 2, (ww - w) // 2 + w, (hh - h) // 2 + h))

            data_dict['image'] = Image.composite(data_dict['image'], bgd, mask)

            if 'mask' in data_dict.keys():
                mask_bgd = Image.new(data_dict['mask'].mode, data_dict['mask'].size, 0)
                data_dict['mask'] = Image.composite(data_dict['mask'], mask_bgd, mask)

        return data_dict

    def __repr__(self):
        return 'GridMask(p={}, use_h={}, use_w={}, ratio={}, rotate={}, offset={}, invert={})'.format(self.p, self.use_h, self.use_w, self.ratio, self.rotate, self.offset, self.invert)

    def rper(self):
        return 'GridMask(not available)'
