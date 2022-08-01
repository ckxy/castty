import torch
import numpy as np
from PIL import Image
from .base_internode import BaseInternode
from torchvision.transforms.functional import to_tensor, to_pil_image
from .builder import INTERNODE
from ..utils.common import is_pil


__all__ = ['ToTensor', 'ToPILImage', 'ToCV2Image']


@INTERNODE.register_module()
class ToTensor(BaseInternode):
    def __call__(self, data_dict):
        assert is_pil(data_dict['image'])

        data_dict['image'] = to_tensor(data_dict['image'])
        
        if 'mask' in data_dict.keys():
            data_dict['mask'] = torch.from_numpy(data_dict['mask'])
        return data_dict

    def reverse(self, **kwargs):
        if 'image' in kwargs.keys():
            kwargs['image'] = to_pil_image(kwargs['image'])
        if 'mask' in kwargs.keys():
            kwargs['mask'] = kwargs['mask'].detach().cpu().numpy().astype(np.int32)
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

