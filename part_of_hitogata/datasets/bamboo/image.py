import random
import numpy as np
from PIL import Image
from .builder import INTERNODE
from ..utils.common import is_pil
from itertools import permutations
from .base_internode import BaseInternode
from torchvision.transforms.functional import normalize


__all__ = ['Normalize', 'SwapChannels', 'RandomSwapChannels']


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

    def forward(self, data_dict):
        data_dict['image'] = normalize(data_dict['image'], self.mean, self.std)
        return data_dict

    def backward(self, data_dict):
        if 'image' in data_dict.keys():
            data_dict['image'] = normalize(data_dict['image'], self.r_mean, self.r_std)
        return data_dict

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

    @staticmethod
    def swap_channels(image, swap):
        if is_pil(image):
            image = np.array(image)
            is_np = False
        else:
            is_np = True

        image = image[..., swap]

        if not is_np:
            image = Image.fromarray(image)
        return image

    def forward(self, data_dict):
        data_dict['image'] = self.swap_channels(data_dict['image'], self.swap)
        return data_dict

    def backward(self, data_dict):
        if 'image' in data_dict.keys():
            data_dict['image'] = self.swap_channels(data_dict['image'], self.r_swap)
        return data_dict

    def __repr__(self):
        return 'SwapChannels(swap={})'.format(self.swap)

    def rper(self):
        return 'SwapChannels(swap={})'.format(self.r_swap)


@INTERNODE.register_module()
class RandomSwapChannels(SwapChannels):
    def __init__(self, **kwargs):
        self.perms = list(permutations(range(3), 3))[1:]

    def calc_intl_param_forward(self, data_dict):
        data_dict['intl_swap'] = random.choice(self.perms)
        return data_dict

    def forward(self, data_dict):
        data_dict['image'] = self.swap_channels(data_dict['image'], data_dict['intl_swap'])
        return data_dict

    def erase_intl_param_forward(self, data_dict):
        data_dict.pop('intl_swap')
        return data_dict

    def backward(self, data_dict):
        return data_dict

    def __repr__(self):
        return type(self).__name__ + '()'

    def rper(self):
        return type(self).__name__ + '(not available)'
