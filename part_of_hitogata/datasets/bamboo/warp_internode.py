import cv2
import numpy as np
from PIL import Image
from .builder import build_internode
from .base_internode import BaseInternode
from ..utils.common import get_image_size, is_pil, clip_bbox, filter_bbox, filter_point, clip_poly, filter_point
from ..utils.warp_tools import fix_cv2_matrix, warp_bbox, warp_mask, warp_point, warp_image, calc_expand_size_and_matrix


__all__ = ['WarpInternode']


base_cfg = [
    dict(type='BboxClipper'),
    dict(type='FilterBboxByLength')
]


class WarpInternode(BaseInternode):
    def __init__(self, expand=False, ccs=False, filter_cfg=base_cfg, **kwargs):
        self.expand = expand
        self.ccs = ccs

        self.filter_internodes = []
        for cfg in filter_cfg:
            self.filter_internodes.append(build_internode(cfg))
        print(self.filter_internodes)
        # exit()

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

        if 'poly' in data_dict.keys():
            data_dict['poly'] = [warp_point(p, M) for p in data_dict['poly']]

        if 'point' in data_dict.keys():
            n = len(data_dict['point'])
            if n > 0:
                points = data_dict['point'].reshape(-1, 2)
                points = warp_point(points, M)
                data_dict['point'] = points.reshape(n, -1, 2)

        if 'mask' in data_dict.keys():
            data_dict['mask'] = warp_mask(data_dict['mask'], M, dst_size, self.ccs)

        data_dict = self.clip_and_filter(data_dict)

        return data_dict

    def clip_and_filter(self, data_dict):
        if 'intl_warp_matrix' in data_dict.keys():
            return data_dict

        # data_dict['clip_size'] = data_dict['intl_warp_tmp_size']
        # for t in self.filter_internodes:
        #     data_dict = t(data_dict)
        # data_dict.pop('clip_size')

        dst_size = data_dict['intl_warp_tmp_size']

        if 'bbox' in data_dict.keys():
            # boxes = data_dict['bbox'].copy()
            data_dict['bbox'] = clip_bbox(data_dict['bbox'], dst_size)
            keep = filter_bbox(data_dict['bbox'])
            data_dict['bbox'] = data_dict['bbox'][keep]

            if 'bbox_meta' in data_dict.keys():
                data_dict['bbox_meta'].filter(keep)

        if 'point' in data_dict.keys():
            n = len(data_dict['point'])
            if n > 0:
                points = data_dict['point'].reshape(-1, 2)

                discard = filter_point(points, dst_size)

                if 'point_meta' in data_dict.keys():
                    visible = data_dict['point_meta']['visible'].reshape(-1)
                    visible[discard] = False
                    data_dict['point_meta']['visible'] = visible.reshape(n, -1)

                data_dict['point'] = points.reshape(n, -1, 2)

        if 'poly' in data_dict.keys():
            data_dict['poly'], keep = clip_poly(data_dict['poly'], dst_size)

            if 'poly_meta' in data_dict.keys():
                data_dict['poly_meta'].filter(keep)

        return data_dict

    def erase_intl_param_forward(self, data_dict):
        data_dict.pop('intl_warp_tmp_matrix')
        data_dict.pop('intl_warp_tmp_size')
        return data_dict

    def __call__(self, data_dict):
        data_dict = self.calc_intl_param_forward(data_dict)
        data_dict = self.forward(data_dict)
        data_dict = self.erase_intl_param_forward(data_dict)
        return data_dict

    def __repr__(self):
        return 'expand={}, ccs={}'.format(self.expand, self.ccs)
