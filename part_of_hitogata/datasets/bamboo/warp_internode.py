import cv2
import numpy as np
from PIL import Image
from .builder import build_internode
from .base_internode import BaseInternode
from .filter_mixin import BaseFilterMixin
from ..utils.common import get_image_size, is_pil, clip_bbox, clip_point, clip_poly
from ..utils.warp_tools import fix_cv2_matrix, warp_bbox, warp_mask, warp_point, warp_image, calc_expand_size_and_matrix


__all__ = ['WarpInternode']


class WarpInternode(BaseInternode, BaseFilterMixin):
    def __init__(self, expand=False, ccs=False, **kwargs):
        self.expand = expand
        self.ccs = ccs

    def calc_intl_param_forward(self, data_dict):
        if 'intl_warp_matrix' in data_dict.keys():
            M = data_dict['intl_warp_tmp_matrix']
            data_dict['intl_warp_matrix'] = M @ data_dict['intl_warp_matrix']
        return data_dict

    def forward(self, data_dict):
        if 'intl_warp_matrix' in data_dict.keys():
            return data_dict

        M = data_dict['intl_warp_tmp_matrix']
        dst_size = data_dict['intl_warp_tmp_size']

        if 'image' in data_dict.keys():
            data_dict['image'] = warp_image(data_dict['image'], M, dst_size, self.ccs)

        if 'bbox' in data_dict.keys():
            data_dict['bbox'] = warp_bbox(data_dict['bbox'], M)
            data_dict['bbox'] = clip_bbox(data_dict['bbox'], dst_size)

        if 'mask' in data_dict.keys():
            data_dict['mask'] = warp_mask(data_dict['mask'], M, dst_size, self.ccs)

        if 'point' in data_dict.keys():
            n = len(data_dict['point'])
            if n > 0:
                points = data_dict['point'].reshape(-1, 2)
                points = warp_point(points, M)
                data_dict['point'] = points.reshape(n, -1, 2)
                data_dict['point'] = clip_point(data_dict['point'], dst_size)

        if 'poly' in data_dict.keys():
            data_dict['poly'] = [warp_point(p, M) for p in data_dict['poly']]
            data_dict['poly'] = clip_poly(data_dict['poly'], dst_size)

        data_dict = self.base_filter(data_dict)

        return data_dict

    def erase_intl_param_forward(self, data_dict):
        data_dict.pop('intl_warp_tmp_matrix')
        data_dict.pop('intl_warp_tmp_size')
        return data_dict

    def __repr__(self):
        return 'expand={}, ccs={}'.format(self.expand, self.ccs)
