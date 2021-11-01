import os
import random
from PIL import Image
from .base_internode import BaseInternode


__all__ = ['RandomFlip']


class RandomFlip(BaseInternode):
    def __init__(self, horizontal=True, mapping=None, p=1):
        assert 0 < p <= 1
        self.p = p
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
        if random.random() < self.p:
            w, h = data_dict['image'].size
            mode = Image.FLIP_LEFT_RIGHT if self.horizontal else Image.FLIP_TOP_BOTTOM

            data_dict['image'] = data_dict['image'].transpose(mode)

            if 'mask' in data_dict.keys():
                data_dict['mask'] = data_dict['mask'].transpose(mode)

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

        return data_dict

    def __repr__(self):
        if self.map_idx is None:
            return 'RandomFlip(horizontal={}, p={})'.format(self.horizontal, self.p)
        else:
            return 'RandomFlip(horizontal={}, p={}, map={})'.format(self.horizontal, self.p, self.map_path)

    def rper(self):
        return 'RandomFlip(not available)'
