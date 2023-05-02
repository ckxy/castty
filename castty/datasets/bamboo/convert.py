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
    def forward_image(self, data_dict):
        target_tag = data_dict['intl_base_target_tag']

        assert is_pil(data_dict[target_tag])
        data_dict[target_tag] = to_tensor(data_dict[target_tag])
        return data_dict

    def forward_mask(self, data_dict):
        target_tag = data_dict['intl_base_target_tag']

        data_dict[target_tag] = torch.from_numpy(data_dict[target_tag])
        return data_dict

    def backward_image(self, data_dict):
        target_tag = data_dict['intl_base_target_tag']

        data_dict[target_tag] = to_pil_image(data_dict[target_tag])
        return data_dict

    def backward_mask(self, data_dict):
        target_tag = data_dict['intl_base_target_tag']

        data_dict[target_tag] = data_dict[target_tag].detach().cpu().numpy().astype(np.int32)
        return data_dict

    def rper(self):
        return 'ToPILImage()'


@INTERNODE.register_module()
class ToPILImage(BaseInternode):
    def forward_image(self, data_dict):
        target_tag = data_dict['intl_base_target_tag']

        assert not is_pil(data_dict[target_tag])
        data_dict[target_tag] = Image.fromarray(data_dict[target_tag])
        return data_dict

    def backward_image(self, data_dict):
        target_tag = data_dict['intl_base_target_tag']

        assert is_pil(data_dict[target_tag])
        data_dict[target_tag] = np.array(data_dict[target_tag])
        return data_dict

    def rper(self):
        return 'ToCV2Image()'


@INTERNODE.register_module()
class ToCV2Image(BaseInternode):
    def forward_image(self, data_dict):
        target_tag = data_dict['intl_base_target_tag']

        assert is_pil(data_dict[target_tag])
        data_dict[target_tag] = np.array(data_dict[target_tag])
        return data_dict

    def backward_image(self, data_dict):
        target_tag = data_dict['intl_base_target_tag']

        assert not is_pil(data_dict[target_tag])
        data_dict[target_tag] = Image.fromarray(data_dict[target_tag])
        return data_dict

    def rper(self):
        return 'ToPILImage()'


@INTERNODE.register_module()
class To1CHTensor(BaseInternode):
    def forward_image(self, data_dict):
        target_tag = data_dict['intl_base_target_tag']

        data_dict[target_tag] = data_dict[target_tag][0].unsqueeze(0)
        return data_dict

    def backward_image(self, data_dict):
        target_tag = data_dict['intl_base_target_tag']
        
        data_dict[target_tag] = data_dict[target_tag].repeat(3, 1, 1)
        return data_dict

    def rper(self):
        return 'To3CHTensor()'
