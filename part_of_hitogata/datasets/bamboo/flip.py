import os
import cv2
import random
from PIL import Image
from .builder import INTERNODE
from .base_internode import BaseInternode
from ..utils.common import get_image_size, is_pil


__all__ = ['Flip']


@INTERNODE.register_module()
class Flip(BaseInternode):
    def __init__(self, horizontal=True, mapping=None, **kwargs):
        self.horizontal = horizontal

        if mapping is None:
            self.map_idx = None
        else:
            with open(os.path.join(mapping), 'r') as f:
                lines = f.readlines()
            assert len(lines) == 1
            map_idx = lines[0].strip().split(',')
            self.map_idx = list(map(int, map_idx))
            self.map_path = mapping
            
        super(Flip, self).__init__(**kwargs)

    def calc_intl_param_forward(self, data_dict):
        w, h = get_image_size(data_dict['image'])
        data_dict['intl_flip_wh'] = (w, h)
        return data_dict

    def forward_image(self, data_dict):
        target_tag = data_dict['intl_base_target_tag']

        if is_pil(data_dict[target_tag]):
            mode = Image.FLIP_LEFT_RIGHT if self.horizontal else Image.FLIP_TOP_BOTTOM
            data_dict[target_tag] = data_dict[target_tag].transpose(mode)
        else:
            mode = 1 if self.horizontal else 0
            data_dict[target_tag] = cv2.flip(data_dict[target_tag], mode)

        return data_dict

    def forward_bbox(self, data_dict):
        w, h = data_dict['intl_flip_wh']
        target_tag = data_dict['intl_base_target_tag']
        
        if self.horizontal:
            data_dict[target_tag][:, 0], data_dict[target_tag][:, 2] = w - data_dict[target_tag][:, 2], w - data_dict[target_tag][:, 0]
        else:
            data_dict[target_tag][:, 1], data_dict[target_tag][:, 3] = h - data_dict[target_tag][:, 3], h - data_dict[target_tag][:, 1]

        return data_dict

    def forward_mask(self, data_dict):
        target_tag = data_dict['intl_base_target_tag']
        
        mode = 1 if self.horizontal else 0
        data_dict[target_tag] = cv2.flip(data_dict[target_tag], mode)

        return data_dict

    def forward_point(self, data_dict):
        w, h = data_dict['intl_flip_wh']
        target_tag = data_dict['intl_base_target_tag']
        
        if self.horizontal:
            data_dict[target_tag][..., 0] = w - data_dict[target_tag][..., 0]
        else:
            data_dict[target_tag][..., 1] = h - data_dict[target_tag][..., 1]
        if self.map_idx is not None:
            for i in range(len(data_dict[target_tag])):
                data_dict[target_tag][i] = data_dict[target_tag][i, self.map_idx]

        return data_dict

    def forward_poly(self, data_dict):
        w, h = data_dict['intl_flip_wh']
        target_tag = data_dict['intl_base_target_tag']
        
        if self.horizontal:
            for i in range(len(data_dict[target_tag])):
                data_dict[target_tag][i][:, 0] = w - data_dict[target_tag][i][:, 0]
        else:
            for i in range(len(data_dict[target_tag])):
                data_dict[target_tag][i][:, 1] = h - data_dict[target_tag][i][:, 1]

        return data_dict

    def erase_intl_param_forward(self, data_dict):
        data_dict.pop('intl_flip_wh')
        return data_dict

    def __repr__(self):
        if self.map_idx is None:
            return 'Flip(horizontal={})'.format(self.horizontal)
        else:
            return 'Flip(horizontal={}, map={})'.format(self.horizontal, self.map_path)
