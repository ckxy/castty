import cv2
import numpy as np
from PIL import Image
from .base_internode import BaseInternode
from .builder import build_internode
from ..utils.warp_tools import get_image_size, fix_cv2_matrix, warp_bbox, clip_bbox, filter_bbox, warp_mask, warp_point, filter_point, warp_image


__all__ = ['WarpInternode']


class WarpInternode(BaseInternode):
    def __init__(self, expand=False, ccs=False, **kwargs):
        self.expand = expand
        self.ccs = ccs

    def __call__(self, data_dict):
        size = get_image_size(data_dict['image'])

        M = data_dict.pop('warp_tmp_matrix')
        dst_size = data_dict.pop('warp_tmp_size')

        if 'warp_matrix' in data_dict.keys():
            data_dict['warp_matrix'] = M @ data_dict['warp_matrix']
            data_dict['warp_size'] = dst_size
        else:
            data_dict['image'] = warp_image(data_dict['image'], M, dst_size, self.ccs)

            if 'bbox' in data_dict.keys():
                boxes = warp_bbox(data_dict['bbox'], M)
                boxes = clip_bbox(boxes, dst_size)
                keep = filter_bbox(boxes)
                data_dict['bbox'] = boxes[keep]

                if 'bbox_meta' in data_dict.keys():
                    data_dict['bbox_meta'].filter(keep)

        return data_dict

    def __repr__(self):
        # return 'WarpInternode()'
        return 'expand={}, ccs={}'.format(self.expand, self.ccs)

    def rper(self):
        return 'WarpInternode(ignore the below internodes)'

