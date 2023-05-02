import cv2
import numpy as np
from PIL import Image
from .builder import build_internode
from .base_internode import BaseInternode
from .mixin import BaseFilterMixin
from ..utils.common import get_image_size, is_pil, clip_bbox, clip_point, clip_poly
from ..utils.warp_tools import fix_cv2_matrix, warp_bbox, warp_mask, warp_point, warp_image, calc_expand_size_and_matrix


__all__ = ['WarpInternode']


class WarpInternode(BaseInternode, BaseFilterMixin):
    def __init__(self, expand=False, ccs=False, **kwargs):
        self.expand = expand
        self.ccs = ccs

        super(WarpInternode, self).__init__(**kwargs)

    def calc_intl_param_forward(self, data_dict):
        if 'intl_warp_matrix' in data_dict.keys():
            M = data_dict['intl_warp_tmp_matrix']
            data_dict['intl_warp_matrix'] = M @ data_dict['intl_warp_matrix']
        return data_dict

    def forward_image(self, data_dict):
        if 'intl_warp_matrix' in data_dict.keys():
            return data_dict
        target_tag = data_dict['intl_base_target_tag']

        M = data_dict['intl_warp_tmp_matrix']
        dst_size = data_dict['intl_warp_tmp_size']

        data_dict[target_tag] = warp_image(data_dict[target_tag], M, dst_size, self.ccs)
        return data_dict

    def forward_bbox(self, data_dict):
        if 'intl_warp_matrix' in data_dict.keys():
            return data_dict
        target_tag = data_dict['intl_base_target_tag']
        
        M = data_dict['intl_warp_tmp_matrix']
        dst_size = data_dict['intl_warp_tmp_size']

        data_dict[target_tag] = warp_bbox(data_dict[target_tag], M)
        data_dict[target_tag] = clip_bbox(data_dict[target_tag], dst_size)

        data_dict = self.base_filter_bbox(data_dict)
        return data_dict

    def forward_mask(self, data_dict):
        if 'intl_warp_matrix' in data_dict.keys():
            return data_dict
        target_tag = data_dict['intl_base_target_tag']

        M = data_dict['intl_warp_tmp_matrix']
        dst_size = data_dict['intl_warp_tmp_size']

        data_dict[target_tag] = warp_mask(data_dict[target_tag], M, dst_size, self.ccs)
        return data_dict

    def forward_point(self, data_dict):
        if 'intl_warp_matrix' in data_dict.keys():
            return data_dict
        target_tag = data_dict['intl_base_target_tag']

        M = data_dict['intl_warp_tmp_matrix']
        dst_size = data_dict['intl_warp_tmp_size']

        n = len(data_dict[target_tag])
        if n > 0:
            points = data_dict[target_tag].reshape(-1, 2)
            points = warp_point(points, M)
            data_dict[target_tag] = points.reshape(n, -1, 2)
            data_dict[target_tag] = clip_point(data_dict[target_tag], dst_size)

        data_dict = self.base_filter_point(data_dict)
        return data_dict

    def forward_poly(self, data_dict):
        if 'intl_warp_matrix' in data_dict.keys():
            return data_dict
        target_tag = data_dict['intl_base_target_tag']

        M = data_dict['intl_warp_tmp_matrix']
        dst_size = data_dict['intl_warp_tmp_size']

        data_dict[target_tag] = [warp_point(p, M) for p in data_dict[target_tag]]
        data_dict[target_tag] = clip_poly(data_dict[target_tag], dst_size)

        data_dict = self.base_filter_poly(data_dict)
        return data_dict

    def erase_intl_param_forward(self, data_dict):
        data_dict.pop('intl_warp_tmp_matrix')
        data_dict.pop('intl_warp_tmp_size')
        return data_dict

    def __repr__(self):
        return 'expand={}, ccs={}'.format(self.expand, self.ccs)
