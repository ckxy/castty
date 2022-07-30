import cv2
import numpy as np
from PIL import Image
from .base_internode import BaseInternode
from .builder import build_internode
from ..utils.warp_tools import fix_cv2_matrix, warp_bbox, warp_mask, warp_point, warp_image
from ..utils.common import get_image_size, is_pil, clip_bbox, filter_bbox, filter_point, clip_poly, filter_point


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

                # print(points, dst_size)
                # print(discard)

                if 'point_meta' in data_dict.keys():
                    ind = data_dict['point_meta'].index('visible')
                    visible = data_dict['point_meta'].values[ind].reshape(-1)
                    visible[discard] = False
                    data_dict['point_meta'].values[ind] = visible.reshape(n, -1)
                    # print(data_dict['point_meta'])
                else:
                    points[discard] = -1

                data_dict['point'] = points.reshape(n, -1, 2)
                # print(data_dict['point'])
                # exit()

        return data_dict

    def __repr__(self):
        # return 'WarpInternode()'
        return 'expand={}, ccs={}'.format(self.expand, self.ccs)

