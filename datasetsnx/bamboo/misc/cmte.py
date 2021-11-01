import random
from PIL import Image
from ..base_internode import BaseInternode
from torchvision.transforms.functional import normalize


__all__ = ['BTurn', 'CMTECrop', 'CMTERandomFlip']


class BTurn(BaseInternode):
    def __init__(self, mean, std):
        self.mean = []
        self.std = []
        for m, s in zip(mean, std):
            self.mean.append(-m / s)
            self.std.append(1 / s)
        self.mean = tuple(self.mean)
        self.std = tuple(self.std)

    def reverse(self, **kwargs):
        if 'b_image' in kwargs.keys():
            kwargs['b_image'] = normalize(kwargs['b_image'], self.mean, self.std)
            kwargs['b_image'] = to_pil_image(kwargs['b_image'])
        return kwargs

    def __repr__(self):
        return 'BTurn(not available)'

    def rper(self):
        return 'BTurn(mean={}, std={})'.format(self.mean, self.std)


class CMTECrop(BaseInternode):
    def __init__(self, size):
        self.size = size
        self.r = 0 < self.size[0] <= 1 and 0 < self.size[1] <= 1

    def __call__(self, data_dict):
        a_img = data_dict['a_image']
        a_star_img = data_dict['a_star_image']
        b_img = data_dict['b_image']

        wast, hast = a_star_img.size
        wb, hb = b_img.size

        if self.r:
            rw = random.randint(0, int(wb * (1 - self.size[0])))
            rh = random.randint(0, int(hb * (1 - self.size[1])))

            b_img = b_img.crop((rw, rh, rw + int(self.size[0] * wb), rh + int(self.size[1] * hb)))
            a_img = a_img.crop((rw, rh, rw + int(self.size[0] * wb), rh + int(self.size[1] * hb)))

            w, h = a_img.size
            rw = random.randint(0, w // 2)
            rh = random.randint(0, h // 2)
            a_img = a_img.crop(rw, rh, rw + w // 2, rh + h // 2)

            rw = random.randint(0, int(wast * (1 - self.size[0] / 2)))
            rh = random.randint(0, int(hast * (1 - self.size[1] / 2)))
            a_star_img = a_star_img.crop((rw, rh, rw + int(self.size[0] * wast / 2), rh + int(self.size[1] * hast / 2)))
        else:
            rw = random.randint(0, wb - self.size[0])
            rh = random.randint(0, hb - self.size[1])

            b_img = b_img.crop((rw, rh, rw + self.size[0], rh + self.size[1]))
            a_img = a_img.crop((rw, rh, rw + self.size[0], rh + self.size[1]))

            w, h = a_img.size
            rw = random.randint(0, w // 2)
            rh = random.randint(0, h // 2)
            a_img = a_img.crop((rw, rh, rw + w // 2, rh + h // 2))

            rw = random.randint(0, wast - self.size[0] // 2)
            rh = random.randint(0, hast - self.size[1] // 2)
            a_star_img = a_star_img.crop((rw, rh, rw + self.size[0] // 2, rh + self.size[1] // 2))

        data_dict['a_image'] = a_img
        data_dict['a_star_image'] = a_star_img
        data_dict['b_image'] = b_img

        return data_dict

    def __repr__(self):
        return 'CMTECrop(size={})'.format(self.size)

    def rper(self):
        return 'CMTECrop(not available)'


class CMTERandomFlip(BaseInternode):
    def __init__(self, horizontal=True, p=1):
        assert 0 < p <= 1
        self.p = p
        self.horizontal = horizontal

    def __call__(self, data_dict):
        if random.random() < self.p:
            mode = Image.FLIP_LEFT_RIGHT if self.horizontal else Image.FLIP_TOP_BOTTOM

            data_dict['a_image'] = data_dict['a_image'].transpose(mode)
            data_dict['a_star_image'] = data_dict['a_star_image'].transpose(mode)
            data_dict['b_image'] = data_dict['b_image'].transpose(mode)

        return data_dict

    def __repr__(self):
        return 'CMTERandomFlip(horizontal={}, p={})'.format(self.horizontal, self.p)

    def rper(self):
        return 'CMTERandomFlip(not available)'
