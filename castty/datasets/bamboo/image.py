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

        super(Normalize, self).__init__(**kwargs)

    def forward_image(self, data_dict):
        target_tag = data_dict['intl_base_target_tag']

        data_dict[target_tag] = normalize(data_dict[target_tag], self.mean, self.std)
        return data_dict

    def backward_image(self, data_dict):
        target_tag = data_dict['intl_base_target_tag']

        data_dict[target_tag] = normalize(data_dict[target_tag], self.r_mean, self.r_std)
        return data_dict

    def __repr__(self):
        return 'Normalize(mean={}, std={})'.format(self.mean, self.std)

    def rper(self):
        return 'Normalize(mean={}, std={})'.format(self.r_mean, self.r_std)


class SwapInternode(BaseInternode):
    def __init__(self, **kwargs):
        super(SwapInternode, self).__init__(**kwargs)

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


@INTERNODE.register_module()
class SwapChannels(SwapInternode):
    def __init__(self, swap, **kwargs):
        self.swap = swap

        self.r_swap = []
        for i in range(len(swap)):
            idx = swap.index(i)
            self.r_swap.append(idx)
        self.r_swap = tuple(self.r_swap)

        super(SwapChannels, self).__init__(**kwargs)

    def forward_image(self, data_dict):
        target_tag = data_dict['intl_base_target_tag']

        data_dict[target_tag] = self.swap_channels(data_dict[target_tag], self.swap)
        return data_dict

    def backward_image(self, data_dict):
        target_tag = data_dict['intl_base_target_tag']

        data_dict[target_tag] = self.swap_channels(data_dict[target_tag], self.r_swap)
        return data_dict

    def __repr__(self):
        return 'SwapChannels(swap={})'.format(self.swap)

    def rper(self):
        return 'SwapChannels(swap={})'.format(self.r_swap)


@INTERNODE.register_module()
class RandomSwapChannels(SwapInternode):
    def __init__(self, **kwargs):
        self.perms = list(permutations(range(3), 3))[1:]

        super(RandomSwapChannels, self).__init__(**kwargs)

    def calc_intl_param_forward(self, data_dict):
        data_dict['intl_swap'] = random.choice(self.perms)
        return data_dict

    def forward_image(self, data_dict):
        target_tag = data_dict['intl_base_target_tag']
        
        data_dict[target_tag] = self.swap_channels(data_dict[target_tag], data_dict['intl_swap'])
        return data_dict

    def erase_intl_param_forward(self, data_dict):
        data_dict.pop('intl_swap')
        return data_dict

    def __repr__(self):
        return type(self).__name__ + '()'

    def rper(self):
        return type(self).__name__ + '(not available)'
