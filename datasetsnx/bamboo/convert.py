import numpy as np
from PIL import Image
from .base_internode import BaseInternode
from torchvision.transforms.functional import to_tensor, to_pil_image


__all__ = ['ToTensor', 'ToPILImage', 'ToCV2Image']


class ToTensor(BaseInternode):
    def __call__(self, data_dict):
        # data_dict = super(ToTensor, self).__call__(data_dict)
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

    def __repr__(self):
        return 'ToTensor()'

    def rper(self):
        return 'ToPILImage()'


class ToPILImage(BaseInternode):
    def __call__(self, data_dict):
        # data_dict = super(ToPILImage, self).__call__(data_dict)
        data_dict['image'] = Image.fromarray(data_dict['image'])
        # if 'mask' in data_dict.keys():
        #     data_dict['mask'] = Image.fromarray(data_dict['mask'])
        return data_dict

    def reverse(self, **kwargs):
        if 'image' in kwargs.keys():
            kwargs['image'] = np.array(kwargs['image'])
        # if 'mask' in kwargs.keys():
        #     kwargs['mask'] = np.array(kwargs['mask'])
        return kwargs

    def __repr__(self):
        return 'ToPILImage()'

    def rper(self):
        return 'ToCV2Image()'


class ToCV2Image(BaseInternode):
    def __call__(self, data_dict):
        # data_dict = super(ToCV2Image, self).__call__(data_dict)
        data_dict['image'] = np.array(data_dict['image'])
        # if 'mask' in data_dict.keys():
        #     data_dict['mask'] = np.array(data_dict['mask'])
        return data_dict

    def reverse(self, **kwargs):
        if 'image' in kwargs.keys():
            kwargs['image'] = Image.fromarray(kwargs['image'])
        # if 'mask' in kwargs.keys():
        #     kwargs['mask'] = Image.fromarray(kwargs['mask'])
        return kwargs

    def __repr__(self):
        return 'ToCV2Image()'

    def rper(self):
        return 'ToPILImage()'

