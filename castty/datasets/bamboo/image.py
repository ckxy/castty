import random
import numpy as np
from PIL import Image
from .builder import INTERNODE
from .mixin import DataAugMixin
from itertools import permutations
from .base_internode import BaseInternode
from ..utils.common import is_pil, is_tensor
from torchvision.transforms.functional import normalize


__all__ = ['Normalize', 'SwapChannels', 'RandomSwapChannels']


@INTERNODE.register_module()
class Normalize(DataAugMixin, BaseInternode):
    def __init__(self, mean, std, tag_mapping=dict(image=['image']), **kwargs):
        self.mean = mean
        self.std = std

        self.r_mean = []
        self.r_std = []
        for m, s in zip(mean, std):
            self.r_mean.append(-m / s)
            self.r_std.append(1 / s)
        self.r_mean = tuple(self.r_mean)
        self.r_std = tuple(self.r_std)

        forward_mapping = dict(
            image=self.forward_image,
        )
        backward_mapping = dict(
            image=self.backward_image,
        )
        # super(Normalize, self).__init__(tag_mapping, forward_mapping, backward_mapping, **kwargs)
        DataAugMixin.__init__(self, tag_mapping, forward_mapping, backward_mapping)
        BaseInternode.__init__(self, **kwargs)

    def forward_image(self, image, meta, **kwargs):
        image = normalize(image, self.mean, self.std)
        return image, meta

    def backward_image(self, image, meta, **kwargs):
        image = normalize(image, self.r_mean, self.r_std)
        return image, meta

    def __repr__(self):
        return 'Normalize(mean={}, std={})'.format(self.mean, self.std)

    def rper(self):
        return 'Normalize(mean={}, std={})'.format(self.r_mean, self.r_std)


class SwapInternode(DataAugMixin, BaseInternode):
    def __init__(self, tag_mapping=dict(image=['image']), **kwargs):
        forward_mapping = dict(
            image=self.forward_image,
        )
        backward_mapping = dict(
            image=self.backward_image,
        )
        # super(SwapInternode, self).__init__(tag_mapping, forward_mapping, backward_mapping, **kwargs)
        DataAugMixin.__init__(self, tag_mapping, forward_mapping, backward_mapping)
        BaseInternode.__init__(self, **kwargs)

    @staticmethod
    def swap_channels(image, swap):
        if is_pil(image):
            image = np.array(image)
            is_np = False
        else:
            is_np = True

        if is_tensor(image):
            image = image[swap, ...]
        else:
            image = image[..., swap]

        if not is_np:
            image = Image.fromarray(image)
        return image

    def forward_image(self, image, meta, **kwargs):
        return image, meta

    def backward_image(self, image, meta, **kwargs):
        return image, meta


@INTERNODE.register_module()
class SwapChannels(SwapInternode):
    def __init__(self, swap, tag_mapping=dict(image=['image']), **kwargs):
        self.swap = swap

        self.r_swap = []
        for i in range(len(swap)):
            idx = swap.index(i)
            self.r_swap.append(idx)
        self.r_swap = tuple(self.r_swap)

        # super(SwapChannels, self).__init__(tag_mapping, **kwargs)
        SwapInternode.__init__(self, tag_mapping, **kwargs)

    def forward_image(self, image, meta, **kwargs):
        image = self.swap_channels(image, self.swap)
        return image, meta

    def backward_image(self, image, meta, **kwargs):
        image = self.swap_channels(image, self.r_swap)
        return image, meta

    def __repr__(self):
        return 'SwapChannels(swap={})'.format(self.swap)

    def rper(self):
        return 'SwapChannels(swap={})'.format(self.r_swap)


@INTERNODE.register_module()
class RandomSwapChannels(SwapInternode):
    def __init__(self, tag_mapping=dict(image=['image']), **kwargs):
        self.perms = list(permutations(range(3), 3))[1:]

        # super(RandomSwapChannels, self).__init__(**kwargs)
        SwapInternode.__init__(self, tag_mapping, **kwargs)

    def calc_intl_param_forward(self, data_dict):
        return dict(intl_swap=random.choice(self.perms))

    def forward_image(self, image, meta, intl_swap, **kwargs):
        image = self.swap_channels(image, intl_swap)
        return image, meta

    def __repr__(self):
        return type(self).__name__ + '()'

    def rper(self):
        return type(self).__name__ + '(not available)'
