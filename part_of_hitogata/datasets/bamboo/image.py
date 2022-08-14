import random
import torchvision
from PIL import Image, ImageFilter
from itertools import permutations
from torch.nn.functional import interpolate as interpolate
from .base_internode import BaseInternode
from torchvision.transforms.functional import normalize, to_grayscale
from .builder import INTERNODE


__all__ = ['Normalize', 'SwapChannels', 'RandomSwapChannels', 'GaussianBlur']


@INTERNODE.register_module()
class Normalize(BaseInternode):
    def __init__(self, mean, std, **kwargs):
        self.mean = mean
        self.std = std

        self.r_mean = []
        self.r_std = []
        for m, s in zip(mean, std):
            self.r_mean.append(-m / s)
            self.r_std.append(1 / s)
        self.r_mean = tuple(self.r_mean)
        self.r_std = tuple(self.r_std)

    def __call__(self, data_dict):
        data_dict['image'] = normalize(data_dict['image'], self.mean, self.std)
        return data_dict

    def reverse(self, **kwargs):
        if 'image' in kwargs.keys():
            kwargs['image'] = normalize(kwargs['image'], self.r_mean, self.r_std)
        return kwargs

    def __repr__(self):
        return 'Normalize(mean={}, std={})'.format(self.mean, self.std)

    def rper(self):
        return 'Normalize(mean={}, std={})'.format(self.r_mean, self.r_std)


@INTERNODE.register_module()
class SwapChannels(BaseInternode):
    def __init__(self, swap, **kwargs):
        self.swap = swap

        self.r_swap = []
        for i in range(len(swap)):
            idx = swap.index(i)
            self.r_swap.append(idx)
        self.r_swap = tuple(self.r_swap)

    def __call__(self, data_dict):
        data_dict['image'] = data_dict['image'][self.swap, ...]
        return data_dict

    def reverse(self, **kwargs):
        if 'image' in kwargs.keys():
            kwargs['image'] = kwargs['image'][self.r_swap, ...]
        return kwargs

    def __repr__(self):
        return 'SwapChannels(swap={})'.format(self.swap)

    def rper(self):
        return 'SwapChannels(swap={})'.format(self.r_swap)


@INTERNODE.register_module()
class RandomSwapChannels(BaseInternode):
    def __init__(self, **kwargs):
        self.perms = list(permutations(range(3), 3))[1:]

    def __call__(self, data_dict):
        swap = random.choice(self.perms)
        data_dict['image'] = data_dict['image'][swap, ...]
        return data_dict


@INTERNODE.register_module()
class GaussianBlur(BaseInternode):
    def __init__(self, radius):
        self.radius = radius

    def __call__(self, data_dict):
        if self.radius <= 0:
            data_dict['image'] = data_dict['image'].filter(ImageFilter.GaussianBlur(radius=random.random()))
        else:
            data_dict['image'] = data_dict['image'].filter(ImageFilter.GaussianBlur(radius=self.radius))
        return data_dict

    def __repr__(self):
        if self.radius <= 0:
            return 'GaussianBlur(random radius)'
        else:
            return 'GaussianBlur(radius={})'.format(self.radius)
