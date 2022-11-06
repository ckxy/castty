import torch
import numpy as np
from PIL import Image
from .builder import INTERNODE
from ..utils.common import is_pil
from .base_internode import BaseInternode
from torchvision.transforms.functional import to_tensor, to_pil_image


__all__ = ['ToTensor', 'ToPILImage', 'ToCV2Image']


@INTERNODE.register_module()
class ToTensor(BaseInternode):
    def forward(self, data_dict):
        if 'image' in data_dict.keys():
            assert is_pil(data_dict['image'])
            data_dict['image'] = to_tensor(data_dict['image'])
        
        if 'mask' in data_dict.keys():
            data_dict['mask'] = torch.from_numpy(data_dict['mask'])
        return data_dict

    def backward(self, data_dict):
        if 'image' in data_dict.keys():
            data_dict['image'] = to_pil_image(data_dict['image'])
        if 'mask' in data_dict.keys():
            data_dict['mask'] = data_dict['mask'].detach().cpu().numpy().astype(np.int32)
        return data_dict

    def rper(self):
        return 'ToPILImage()'


@INTERNODE.register_module()
class ToPILImage(BaseInternode):
    def forward(self, data_dict):
        if 'image' in data_dict.keys():
            assert not is_pil(data_dict['image'])
            data_dict['image'] = Image.fromarray(data_dict['image'])
        return data_dict

    def backward(self, data_dict):
        if 'image' in data_dict.keys():
            assert is_pil(data_dict['image'])
            data_dict['image'] = np.array(data_dict['image'])
        return data_dict

    def rper(self):
        return 'ToCV2Image()'


@INTERNODE.register_module()
class ToCV2Image(BaseInternode):
    def forward(self, data_dict):
        if 'image' in data_dict.keys():
            assert is_pil(data_dict['image'])
            data_dict['image'] = np.array(data_dict['image'])
        return data_dict

    def backward(self, data_dict):
        if 'image' in data_dict.keys():
            assert not is_pil(data_dict['image'])
            data_dict['image'] = Image.fromarray(data_dict['image'])
        return data_dict

    def rper(self):
        return 'ToPILImage()'


@INTERNODE.register_module()
class To1CHTensor(BaseInternode):
    def forward(self, data_dict):
        if 'image' in data_dict.keys():
            data_dict['image'] = data_dict['image'][0].unsqueeze(0)
        return data_dict

    def backward(self, data_dict):
        if 'image' in data_dict.keys():
            data_dict['image'] = data_dict['image'].repeat(3, 1, 1)
        return data_dict

    def rper(self):
        return 'To3CHTensor()'
