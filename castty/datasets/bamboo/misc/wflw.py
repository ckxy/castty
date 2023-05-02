import random
import numpy as np
from ..builder import INTERNODE
from ..crop import CropInternode
from ...utils.common import get_image_size
from ..base_internode import BaseInternode
from ....utils.bbox_tools import xyxy2xywh, xywh2xyxy


__all__ = ['WFLWCrop']


@INTERNODE.register_module()
class WFLWCrop(CropInternode):
    def __init__(self, mode='point', expand=(1, 1), return_offset=False, **kwargs):
        assert mode in ['point', 'box']
        self.mode = mode
        self.expand = expand
        self.return_offset = return_offset

        super(WFLWCrop, self).__init__(use_base_filter=False, **kwargs)

    def calc_cropping(self, data_dict):
        box2point = data_dict.pop('bbox_meta')['box2point'].tolist()
        idx = random.choice(box2point)

        if self.mode == 'point':
            points = data_dict['point'][idx]
            box = np.concatenate((np.min(points, axis=0), np.max(points, axis=0)))
        else:
            box = data_dict['bbox'][idx]
        data_dict.pop('bbox')

        r = random.uniform(self.expand[0], self.expand[1])
        w, h = get_image_size(data_dict['image'])

        box = xyxy2xywh(box)
        box[2:] *= r
        x1, y1, x2, y2 = np.around(xywh2xyxy(box)).astype(np.int32)

        return x1, y1, x2, y2

    def forward(self, data_dict):
        data_dict = super(WFLWCrop, self).forward(data_dict)

        if self.return_offset:
            data_dict['wflw_offset'] = box[:2].astype(np.float32)

        return data_dict

    def backward(self, data_dict):
        return data_dict

    def __repr__(self):
        return 'WFLWCrop(mode={}, expand={}, return_offset={})'.format(self.mode, self.expand, self.return_offset)
