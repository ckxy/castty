import cv2
import numpy as np
from PIL import Image
from .builder import build_internode
from .base_internode import BaseInternode
from ..utils.common import get_image_size, is_pil, clip_bbox, filter_bbox, filter_point, clip_poly, filter_point
from ..utils.warp_tools import fix_cv2_matrix, warp_bbox, warp_mask, warp_point, warp_image, calc_expand_size_and_matrix


__all__ = ['WarpInternode']


class WarpInternode(BaseInternode):
    def __init__(self, expand=False, ccs=False, **kwargs):
        self.expand = expand
        self.ccs = ccs

    def calc_intl_param_forward(self, data_dict):
        if 'warp_matrix' in data_dict.keys():
            M = data_dict['warp_tmp_matrix']
            # dst_size = data_dict['warp_tmp_size']

            data_dict['warp_matrix'] = M @ data_dict['warp_matrix']
            # data_dict['warp_size'] = dst_size

            # if self.expand:
            #     E, new_size = calc_expand_size_and_matrix(M, dst_size)
            #     data_dict['warp_matrix'] = E @ data_dict['warp_matrix']
            #     data_dict['warp_size'] = new_size
            # else:
            #     data_dict['warp_size'] = dst_size
        return data_dict

    def forward(self, data_dict):
        if 'warp_matrix' in data_dict.keys():
            return data_dict

        M = data_dict['warp_tmp_matrix']
        dst_size = data_dict['warp_tmp_size']

        if 'image' in data_dict.keys():
            data_dict['image'] = warp_image(data_dict['image'], M, dst_size, self.ccs)

        if 'bbox' in data_dict.keys():
            boxes = warp_bbox(data_dict['bbox'], M)
            boxes = clip_bbox(boxes, dst_size)
            keep = filter_bbox(boxes)
            data_dict['bbox'] = boxes[keep]

            if 'bbox_meta' in data_dict.keys():
                data_dict['bbox_meta'].filter(keep)

        if 'poly' in data_dict.keys():
            data_dict['poly'] = [warp_point(p, M) for p in data_dict['poly']]
            data_dict['poly'], keep = clip_poly(data_dict['poly'], dst_size)

            if 'poly_meta' in data_dict.keys():
                data_dict['poly_meta'].filter(keep)

        if 'point' in data_dict.keys():
            n = len(data_dict['point'])
            points = data_dict['point'].reshape(-1, 2)
            points = warp_point(points, M)

            discard = filter_point(points, dst_size)

            if 'point_meta' in data_dict.keys():
                visible = data_dict['point_meta']['visible'].reshape(-1)
                visible[discard] = False
                data_dict['point_meta']['visible'] = visible.reshape(n, -1)
            else:
                points[discard] = -1

            data_dict['point'] = points.reshape(n, -1, 2)

        if 'mask' in data_dict.keys():
            data_dict['mask'] = warp_mask(data_dict['mask'], M, dst_size, self.ccs)
        return data_dict

    def erase_intl_param_forward(self, data_dict):
        data_dict.pop('warp_tmp_matrix')
        data_dict.pop('warp_tmp_size')
        return data_dict

    def __call__(self, data_dict):
        # M = data_dict.pop('warp_tmp_matrix')
        # dst_size = data_dict.pop('warp_tmp_size')

        # if 'warp_matrix' in data_dict.keys():
        #     data_dict['warp_matrix'] = M @ data_dict['warp_matrix']
        #     data_dict['warp_size'] = dst_size
        # else:
        #     data_dict['image'] = warp_image(data_dict['image'], M, dst_size, self.ccs)

        #     if 'bbox' in data_dict.keys():
        #         boxes = warp_bbox(data_dict['bbox'], M)
        #         boxes = clip_bbox(boxes, dst_size)
        #         keep = filter_bbox(boxes)
        #         data_dict['bbox'] = boxes[keep]

        #         if 'bbox_meta' in data_dict.keys():
        #             data_dict['bbox_meta'].filter(keep)

        #     if 'poly' in data_dict.keys():
        #         data_dict['poly'] = [warp_point(p, M) for p in data_dict['poly']]
        #         data_dict['poly'], keep = clip_poly(data_dict['poly'], dst_size)

        #         if 'poly_meta' in data_dict.keys():
        #             data_dict['poly_meta'].filter(keep)

        #     if 'point' in data_dict.keys():
        #         n = len(data_dict['point'])
        #         points = data_dict['point'].reshape(-1, 2)
        #         points = warp_point(points, M)

        #         discard = filter_point(points, dst_size)

        #         if 'point_meta' in data_dict.keys():
        #             visible = data_dict['point_meta']['visible'].reshape(-1)
        #             visible[discard] = False
        #             data_dict['point_meta']['visible'] = visible.reshape(n, -1)
        #         else:
        #             points[discard] = -1

        #         data_dict['point'] = points.reshape(n, -1, 2)

        #     if 'mask' in data_dict.keys():
        #         data_dict['mask'] = warp_mask(data_dict['mask'], M, dst_size, self.ccs)

        data_dict = self.calc_intl_param_forward(data_dict)
        data_dict = self.forward(data_dict)
        data_dict = self.erase_intl_param_forward(data_dict)
        return data_dict

    def __repr__(self):
        return 'expand={}, ccs={}'.format(self.expand, self.ccs)

