import random
import numpy as np
from ..base_internode import BaseInternode
from utils.bbox_tools import xyxy2xywh, xywh2xyxy


__all__ = ['CropROKP']


class CropROKP(BaseInternode):
    def __init__(self, mode, expand=(1, 1), return_offset=False):
        assert mode in ['point', 'box']
        self.mode = mode
        self.expand = expand
        self.return_offset = return_offset

    def __call__(self, data_dict):
        if self.mode == 'point':
            box = np.concatenate((np.min(data_dict['point'], axis=0), np.max(data_dict['point'], axis=0)))
        else:
            box = data_dict['bbox'][0]

        r = random.uniform(self.expand[0], self.expand[1])
        w, h = data_dict['image'].size

        box = xyxy2xywh(box)
        box[2:] *= r
        x1, y1, x2, y2 = np.around(xywh2xyxy(box)).astype(np.int)

        data_dict['image'] = data_dict['image'].crop((x1, y1, x2, y2))
        data_dict['ori_size'] = np.array((data_dict['image'].size[1], data_dict['image'].size[0])).astype(np.float32)

        if 'point' in data_dict.keys():
            data_dict['point'][..., 0] -= box[0]
            data_dict['point'][..., 1] -= box[1]

        if self.return_offset:
            data_dict['offset'] = box[:2].astype(np.float32)

        return data_dict

    def __repr__(self):
        return 'CropROKP(mode={}, expand={}, return_offset={})'.format(self.mode, self.expand, self.return_offset)

    def rper(self):
        return 'CropROKP(not available)'
