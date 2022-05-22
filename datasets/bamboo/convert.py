import numpy as np
from PIL import Image
from .base_internode import BaseInternode
from torchvision.transforms.functional import to_tensor, to_pil_image
from .builder import INTERNODE


__all__ = ['ToTensor', 'ToPILImage', 'ToCV2Image']


@INTERNODE.register_module()
class ToTensor(BaseInternode):
    def __call__(self, data_dict):
        data_dict['image'] = to_tensor(data_dict['image'])
        if 'mask' in data_dict.keys():
            data_dict['mask'] = (to_tensor(data_dict['mask']) * 255).long().squeeze()
        return data_dict

    def reverse(self, **kwargs):
        if 'image' in kwargs.keys():
            kwargs['image'] = to_pil_image(kwargs['image'])
        if 'mask' in kwargs.keys():
            kwargs['mask'] = to_pil_image(kwargs['mask'].float() / 255)
        return kwargs

    def rper(self):
        return 'ToPILImage()'


@INTERNODE.register_module()
class ToPILImage(BaseInternode):
    def __call__(self, data_dict):
        data_dict['image'] = Image.fromarray(data_dict['image'])
        return data_dict

    def reverse(self, **kwargs):
        if 'image' in kwargs.keys():
            kwargs['image'] = np.array(kwargs['image'])
        return kwargs

    def rper(self):
        return 'ToCV2Image()'


@INTERNODE.register_module()
class ToCV2Image(BaseInternode):
    def __call__(self, data_dict):
        data_dict['image'] = np.array(data_dict['image'])
        return data_dict

    def reverse(self, **kwargs):
        if 'image' in kwargs.keys():
            kwargs['image'] = Image.fromarray(kwargs['image'])
        return kwargs

    def rper(self):
        return 'ToPILImage()'

