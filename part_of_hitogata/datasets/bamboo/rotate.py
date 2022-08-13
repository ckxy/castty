import cv2
import math
import random
import numpy as np
from PIL import Image
from .builder import INTERNODE
from .base_internode import BaseInternode
from ..utils.warp_tools import calc_expand_size_and_matrix, warp_bbox, warp_point, warp_mask
from ..utils.common import get_image_size, is_pil


__all__ = ['Rot90']


@INTERNODE.register_module()
class Rot90(BaseInternode):
    def __init__(self, k=[0, 1, 2, 3], **kwargs):
        assert set(k) | set([0, 1, 2, 3]) == set([0, 1, 2, 3])
        self.k = sorted(list(k))

    @staticmethod
    def build_matrix(angle, img_size):
        w, h = img_size
        angle = math.radians(angle)

        C = np.eye(3)
        C[0, 2] = -w / 2
        C[1, 2] = -h / 2

        R = np.eye(3)
        R[0, 0] = round(math.cos(angle), 15)
        R[0, 1] = -round(math.sin(angle), 15)
        R[1, 0] = round(math.sin(angle), 15)
        R[1, 1] = round(math.cos(angle), 15)

        CI = np.eye(3)
        CI[0, 2] = w / 2
        CI[1, 2] = h / 2

        return CI @ R @ C

    def __call__(self, data_dict):
        angle = random.choice(self.k) * 90

        # angle = 90
        # print(angle)

        if angle != 0:
            size = get_image_size(data_dict['image'])
            M = self.build_matrix(angle, size)

            E, size = calc_expand_size_and_matrix(M, size)
            M = E @ M

            if is_pil(data_dict['image']):
                if angle == 90:
                    data_dict['image'] = data_dict['image'].transpose(Image.Transpose.ROTATE_270)
                elif angle == 180:
                    data_dict['image'] = data_dict['image'].transpose(Image.Transpose.ROTATE_180)
                else:
                    data_dict['image'] = data_dict['image'].transpose(Image.Transpose.ROTATE_90)
            else:
                if angle == 90:
                    data_dict['image'] = cv2.rotate(data_dict['image'], cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    data_dict['image'] = cv2.rotate(data_dict['image'], cv2.ROTATE_180)
                else:
                    data_dict['image'] = cv2.rotate(data_dict['image'], cv2.ROTATE_90_COUNTERCLOCKWISE)

            if 'bbox' in data_dict.keys():
                boxes = warp_bbox(data_dict['bbox'], M)

            if 'poly' in data_dict.keys():
                data_dict['poly'] = [warp_point(p, M) for p in data_dict['poly']]

            if 'point' in data_dict.keys():
                n = len(data_dict['point'])
                points = data_dict['point'].reshape(-1, 2)
                points = warp_point(points, M)

            if 'mask' in data_dict.keys():
                data_dict['mask'] = warp_mask(data_dict['mask'], M, dst_size, True)

        return data_dict

    def __repr__(self):
        return 'Rot90(k={})'.format(tuple(self.k))
