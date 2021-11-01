import cv2
import numpy as np
from PIL import Image
from .base_internode import BaseInternode
from .utils.warp_tools import get_image_mode_and_size, fix_cv2_matrix, warp_bbox, clip_bbox, filter_bbox, warp_mask, warp_point, filter_point


__all__ = ['WarpInternode']


class WarpInternode(BaseInternode):
    def __init__(self, expand=False, ccs=True, p=1):
        assert 0 < p <= 1

        # self.out_m = kwargs['out_m'] if 'out_m' in kwargs.keys() else False
        self.expand = expand
        self.ccs = ccs
        self.p = p

        # self.filter = dict(
        #     bbox=
        # )

    def __call__(self, data_dict):
        use_pil, size = get_image_mode_and_size(data_dict['image'])
        M = data_dict.pop('warp_tmp_matrix')
        dst_size = data_dict.pop('warp_tmp_size')

        if 'warp_matrix' in data_dict.keys():
            data_dict['warp_matrix'] = M @ data_dict['warp_matrix']
            data_dict['warp_size'] = dst_size
        else:
            if use_pil:
                matrix = np.array(np.matrix(M).I).flatten()
                matrix = (matrix / matrix[-1]).tolist()
                data_dict['image'] = data_dict['image'].transform(dst_size, Image.PERSPECTIVE, matrix, Image.BILINEAR)
            else:
                matrix = np.matrix(M)
                if self.ccs:
                    matrix = fix_cv2_matrix(matrix)
                # print(matrix)
                data_dict['image'] = cv2.warpPerspective(data_dict['image'], matrix, dst_size)

            if 'bbox' in data_dict.keys():
                boxes = data_dict['bbox'][:, :4]
                other = data_dict['bbox'][:, 4:]
                boxes = warp_bbox(boxes, M)
                boxes = clip_bbox(boxes, dst_size)
                keep = filter_bbox(boxes)

                data_dict['bbox'] = np.concatenate((boxes, other), axis=-1)[keep]
                if 'difficult' in data_dict.keys():
                    data_dict['difficult'] = data_dict['difficult'][keep]
                # print(data_dict['bbox'], keep)

            if 'point' in data_dict.keys():
                points = warp_point(data_dict['point'], M)
                discard = filter_point(points, dst_size)

                if 'visible' in data_dict.keys():
                    data_dict['visible'][discard] = 0
                else:
                    points[discard] = -1
                # print(data_dict.keys())
                # exit()
                data_dict['point'] = points

            if 'mask' in data_dict.keys():
                data_dict['mask'] = warp_mask(data_dict['mask'], M, dst_size)
        return data_dict

    def __repr__(self):
        # return 'WarpInternode()'
        return 'p={}, expand={}, ccs={}'.format(self.p, self.expand, self.ccs)

    def rper(self):
        return 'WarpInternode(ignore the below internodes)'
