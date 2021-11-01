import random
import torchvision
from PIL import Image, ImageFilter
from itertools import permutations
from torch.nn.functional import interpolate as interpolate
from .base_internode import BaseInternode
from torchvision.transforms.functional import normalize, to_grayscale


__all__ = ['Normalize', 'SwapChannels', 'RandomSwapChannels', 'GaussianBlur']


class Normalize(BaseInternode):
    def __init__(self, mean, std):
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


class SwapChannels(BaseInternode):
    def __init__(self, swap):
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


class RandomSwapChannels(BaseInternode):
    def __init__(self, ch=3, p=1):
        assert 0 < p <= 1 and ch > 1
        self.p = p
        self.ch = ch
        self.perms = list(permutations(range(self.ch), self.ch))[1:]

    def __call__(self, data_dict):
        if random.random() < self.p:
            swap = random.choice(self.perms)
            data_dict['image'] = data_dict['image'][swap, ...]
        return data_dict

    def __repr__(self):
        return 'RandomSwapChannels(p={}, ch={})'.format(self.p, self.ch)

    def rper(self):
        return 'RandomSwapChannels(not available)'


# class MultiScaleTest(BaseInternode):
#     def __init__(self, **kwargs):
#         super(MultiScaleTest, self).__init__(**kwargs)
#         if self.is_reverse:
#             self.ori_size = kwargs['ori_size']
#             return

#         self.sizes = kwargs['sizes']
#         assert len(self.sizes) > 1
#         self.sizes = tuple(sorted(self.sizes))

#     def __call__(self, data_dict):
#         data_dict['test_sizes'] = self.sizes
#         return data_dict

#     def reverse(self, **kwargs):
#         # print(kwargs.keys())
#         s = None
#         if 'test_size' in kwargs.keys():
#             s = kwargs['test_size']
#         if 'image' in kwargs.keys():
#             s = kwargs['image'].shape[-1]
#         assert s is not None and isinstance(s, int)
        
#         if 'image' in kwargs.keys():
#             kwargs['image'] = interpolate(kwargs['image'].unsqueeze(0), size=(self.ori_size, self.ori_size), mode='bilinear', align_corners=False)[0]
#         if 'bbox' in kwargs.keys():
#             kwargs['bbox'][:, :4] = kwargs['bbox'][:, :4] / s * self.ori_size
#         #     print(kwargs['bbox'][:, :4])
#         # exit()

#         return kwargs

#     def __repr__(self):
#         if self.is_reverse:
#             return 'MultiScaleTest(ori_size={})'.format(self.ori_size)
#         else:
#             return 'MultiScaleTest(sizes={})'.format(self.sizes)


class GaussianBlur(BaseInternode):
    def __init__(self, radius, p=1):
        assert 0 < p <= 1
        self.p = p
        self.radius = radius

    def __call__(self, data_dict):
        if random.random() < self.p:
            if self.radius <= 0:
                data_dict['image'] = data_dict['image'].filter(ImageFilter.GaussianBlur(radius=random.random()))
            else:
                data_dict['image'] = data_dict['image'].filter(ImageFilter.GaussianBlur(radius=self.radius))
        return data_dict

    def __repr__(self):
        if self.radius <= 0:
            return 'GaussianBlur(p={}, random radius)'.format(self.p)
        else:
            return 'GaussianBlur(p={}, radius={})'.format(self.p, self.radius)

    def rper(self):
        return 'GaussianBlur(not available)'


# class BlendingWithMask(BaseInternode):
#     def __init__(self, **kwargs):
#         super(BlendingWithMask, self).__init__(**kwargs)
#         if self.is_reverse:
#             return

#         self.alpha = kwargs['alpha']
#         assert 0 < self.alpha < 1

#     def __call__(self, data_dict):
#         alpha = random.uniform(0, self.alpha)
#         mask = Image.new(data_dict['image'].mode, data_dict['image'].size, 0)
#         data_dict['image'] = Image.blend(data_dict['image'], mask, alpha)
#         return data_dict

#     def __repr__(self):
#         if self.is_reverse:
#             return 'BlendingWithMask(not available)'
#         else:
#             return 'BlendingWithMask(alpha={})'.format(self.alpha)
