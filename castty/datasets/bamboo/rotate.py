import cv2
import math
import random
import numpy as np
from PIL import Image
from .builder import INTERNODE
from .mixin import DataAugMixin
from .base_internode import BaseInternode
from ..utils.common import get_image_size, is_pil, is_cv2
from torchvision.transforms import functional, InterpolationMode
from ..utils.warp_tools import calc_expand_size_and_matrix, warp_bbox, warp_point


__all__ = ['Rot90']


TAG_MAPPING = dict(
    image=['image'],
    bbox=['bbox'],
    mask=['mask'],
    point=['point'],
    poly=['poly'],
)


@INTERNODE.register_module()
class Rot90(DataAugMixin, BaseInternode):
    def __init__(self, k=[1, 2, 3], tag_mapping=TAG_MAPPING, **kwargs):
        assert set(k) | set([1, 2, 3]) == set([1, 2, 3])
        self.k = sorted(list(k))

        forward_mapping = dict(
            image=self.forward_image,
            bbox=self.forward_bbox,
            mask=self.forward_mask,
            point=self.forward_point,
            poly=self.forward_poly
        )
        backward_mapping = dict()
        # super(Rot90, self).__init__(tag_mapping, forward_mapping, backward_mapping, **kwargs)
        DataAugMixin.__init__(self, tag_mapping, forward_mapping, backward_mapping)
        BaseInternode.__init__(self, **kwargs)

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
        param = dict()
        param['intl_rot90_angle'] = random.choice(self.k) * 90

        if param['intl_rot90_angle'] != 0:
            size = get_image_size(data_dict['image'])
            M = self.build_matrix(param['intl_rot90_angle'], size)

            E, _ = calc_expand_size_and_matrix(M, size)
            param['intl_rot90_matrix'] = E @ M
        else:
            param['intl_rot90_matrix'] = None

        return param

    def forward_image(self, image, meta, intl_rot90_angle, intl_rot90_matrix, **kwargs):
        if intl_rot90_angle == 0:
            return image, meta

        if is_pil(image):
            if intl_rot90_angle == 90:
                image = image.transpose(Image.Transpose.ROTATE_270)
            elif intl_rot90_angle == 180:
                image = image.transpose(Image.Transpose.ROTATE_180)
            else:
                image = image.transpose(Image.Transpose.ROTATE_90)
        elif is_cv2(image):
            if intl_rot90_angle == 90:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif intl_rot90_angle == 180:
                image = cv2.rotate(image, cv2.ROTATE_180)
            else:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            image = functional.rotate(image, 360 - intl_rot90_angle, InterpolationMode.NEAREST, True, None, 0)
        return image, meta

    def forward_bbox(self, bbox, meta, intl_rot90_angle, intl_rot90_matrix, **kwargs):
        if intl_rot90_angle == 0:
            return bbox, meta        
        bbox = warp_bbox(bbox, intl_rot90_matrix)
        return bbox, meta

    def forward_mask(self, mask, meta, intl_rot90_angle, intl_rot90_matrix, **kwargs):
        if intl_rot90_angle == 0:
            return mask, meta

        if is_cv2(mask):
            if intl_rot90_angle == 90:
                mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
            elif intl_rot90_angle == 180:
                mask = cv2.rotate(mask, cv2.ROTATE_180)
            else:
                mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            mask = mask.unsqueeze(0)
            mask = functional.rotate(mask, 360 - intl_rot90_angle, InterpolationMode.NEAREST, True, None, 0)
            mask = mask[0]
        return mask, meta

    def forward_point(self, point, meta, intl_rot90_angle, intl_rot90_matrix, **kwargs):
        if intl_rot90_angle == 0:
            return point, meta
        
        n = len(point)
        if n > 0:
            point = point.reshape(-1, 2)
            point = warp_point(point, intl_rot90_matrix)
            point = point.reshape(n, -1, 2)
        return point, meta

    def forward_poly(self, poly, meta, intl_rot90_angle, intl_rot90_matrix, **kwargs):
        if intl_rot90_angle == 0:
            return poly, meta
        poly = [warp_point(p, intl_rot90_matrix) for p in poly]
        return poly, meta

    def __repr__(self):
        return 'Rot90(k={})'.format(tuple(self.k))
