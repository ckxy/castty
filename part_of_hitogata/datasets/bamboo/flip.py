import os
import cv2
import random
from PIL import Image
from .base_internode import BaseInternode
from .builder import INTERNODE
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

    def __call__(self, data_dict):
        w, h = get_image_size(data_dict['image'])

        if is_pil(data_dict['image']):
            mode = Image.FLIP_LEFT_RIGHT if self.horizontal else Image.FLIP_TOP_BOTTOM
            data_dict['image'] = data_dict['image'].transpose(mode)
        else:
            mode = 1 if self.horizontal else 0
            data_dict['image'] = cv2.flip(data_dict['image'], mode)

        if 'bbox' in data_dict.keys():
            if self.horizontal:
                data_dict['bbox'][:, 0], data_dict['bbox'][:, 2] = w - data_dict['bbox'][:, 2], w - data_dict['bbox'][:, 0]
            else:
                data_dict['bbox'][:, 1], data_dict['bbox'][:, 3] = h - data_dict['bbox'][:, 3], h - data_dict['bbox'][:, 1]

        if 'point' in data_dict.keys():
            if self.horizontal:
                data_dict['point'][:, 0] = w - data_dict['point'][:, 0]
            else:
                data_dict['point'][:, 1] = h - data_dict['point'][:, 1]
            if self.map_idx is not None:
                data_dict['point'] = data_dict['point'][self.map_idx]

        if 'poly' in data_dict.keys():
            if self.horizontal:
                for i in range(len(data_dict['poly'])):
                    data_dict['poly'][i][:, 0] = w - data_dict['poly'][i][:, 0]
            else:
                for i in range(len(data_dict['poly'])):
                    data_dict['poly'][i][:, 1] = h - data_dict['poly'][i][:, 1]

        return data_dict

    def __repr__(self):
        if self.map_idx is None:
            return 'Flip(horizontal={})'.format(self.horizontal)
        else:
            return 'Flip(horizontal={}, map={})'.format(self.horizontal, self.map_path)
