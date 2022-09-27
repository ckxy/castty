import cv2
import math
import random
import numpy as np
from PIL import Image
from .builder import INTERNODE
from .base_internode import BaseInternode
from ..utils.common import get_image_size, is_pil
from ..utils.warp_tools import calc_expand_size_and_matrix, warp_bbox, warp_point


__all__ = ['Rot90']


@INTERNODE.register_module()
class Rot90(BaseInternode):
    def __init__(self, k=[1, 2, 3], **kwargs):
        assert set(k) | set([1, 2, 3]) == set([1, 2, 3])
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

    def calc_intl_param_forward(self, data_dict):
        data_dict['intl_rot90_angle'] = random.choice(self.k) * 90

        if data_dict['intl_rot90_angle'] != 0:
            size = get_image_size(data_dict['image'])
            M = self.build_matrix(data_dict['intl_rot90_angle'], size)

            E, _ = calc_expand_size_and_matrix(M, size)
            data_dict['intl_rot90_matrix'] = E @ M
        else:
            data_dict['intl_rot90_matrix'] = None

        return data_dict

    def forward(self, data_dict):
        if data_dict['intl_rot90_angle'] == 0:
            return data_dict

        if 'image' in data_dict.keys():
            if is_pil(data_dict['image']):
                if data_dict['intl_rot90_angle'] == 90:
                    data_dict['image'] = data_dict['image'].transpose(Image.Transpose.ROTATE_270)
                elif data_dict['intl_rot90_angle'] == 180:
                    data_dict['image'] = data_dict['image'].transpose(Image.Transpose.ROTATE_180)
                else:
                    data_dict['image'] = data_dict['image'].transpose(Image.Transpose.ROTATE_90)
            else:
                if data_dict['intl_rot90_angle'] == 90:
                    data_dict['image'] = cv2.rotate(data_dict['image'], cv2.ROTATE_90_CLOCKWISE)
                elif data_dict['intl_rot90_angle'] == 180:
                    data_dict['image'] = cv2.rotate(data_dict['image'], cv2.ROTATE_180)
                else:
                    data_dict['image'] = cv2.rotate(data_dict['image'], cv2.ROTATE_90_COUNTERCLOCKWISE)

        if 'bbox' in data_dict.keys():
            boxes = warp_bbox(data_dict['bbox'], data_dict['intl_rot90_matrix'])

        if 'poly' in data_dict.keys():
            data_dict['poly'] = [warp_point(p, data_dict['intl_rot90_matrix']) for p in data_dict['poly']]

        if 'point' in data_dict.keys():
            n = len(data_dict['point'])
            points = data_dict['point'].reshape(-1, 2)
            points = warp_point(points, data_dict['intl_rot90_matrix'])

        if 'mask' in data_dict.keys():
            if data_dict['intl_rot90_angle'] == 90:
                data_dict['mask'] = cv2.rotate(data_dict['mask'], cv2.ROTATE_90_CLOCKWISE)
            elif data_dict['intl_rot90_angle'] == 180:
                data_dict['mask'] = cv2.rotate(data_dict['mask'], cv2.ROTATE_180)
            else:
                data_dict['mask'] = cv2.rotate(data_dict['mask'], cv2.ROTATE_90_COUNTERCLOCKWISE)

        return data_dict

    def erase_intl_param_forward(self, data_dict):
        data_dict.pop('intl_rot90_angle')
        data_dict.pop('intl_rot90_matrix')
        return data_dict

    def __call__(self, data_dict):
        # angle = random.choice(self.k) * 90

        # if angle != 0:
        #     size = get_image_size(data_dict['image'])
        #     M = self.build_matrix(angle, size)

        #     E, size = calc_expand_size_and_matrix(M, size)
        #     M = E @ M

        #     if is_pil(data_dict['image']):
        #         if angle == 90:
        #             data_dict['image'] = data_dict['image'].transpose(Image.Transpose.ROTATE_270)
        #         elif angle == 180:
        #             data_dict['image'] = data_dict['image'].transpose(Image.Transpose.ROTATE_180)
        #         else:
        #             data_dict['image'] = data_dict['image'].transpose(Image.Transpose.ROTATE_90)
        #     else:
        #         if angle == 90:
        #             data_dict['image'] = cv2.rotate(data_dict['image'], cv2.ROTATE_90_CLOCKWISE)
        #         elif angle == 180:
        #             data_dict['image'] = cv2.rotate(data_dict['image'], cv2.ROTATE_180)
        #         else:
        #             data_dict['image'] = cv2.rotate(data_dict['image'], cv2.ROTATE_90_COUNTERCLOCKWISE)

        #     if 'bbox' in data_dict.keys():
        #         boxes = warp_bbox(data_dict['bbox'], M)

        #     if 'poly' in data_dict.keys():
        #         data_dict['poly'] = [warp_point(p, M) for p in data_dict['poly']]

        #     if 'point' in data_dict.keys():
        #         n = len(data_dict['point'])
        #         points = data_dict['point'].reshape(-1, 2)
        #         points = warp_point(points, M)

        #     if 'mask' in data_dict.keys():
        #         data_dict['mask'] = warp_mask(data_dict['mask'], M, dst_size, True)
        data_dict = self.calc_intl_param_forward(data_dict)
        data_dict = self.forward(data_dict)
        data_dict = self.erase_intl_param_forward(data_dict)
        return data_dict

    def __repr__(self):
        return 'Rot90(k={})'.format(tuple(self.k))
