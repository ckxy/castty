import cv2
import numpy as np
from PIL import Image
from .builder import build_internode
from .base_internode import BaseInternode
from .mixin import BaseFilterMixin, DataAugMixin
from ..utils.common import get_image_size, is_pil, clip_bbox, clip_point, clip_poly
from ..utils.warp_tools import fix_cv2_matrix, warp_bbox, warp_mask, warp_point, warp_image, calc_expand_size_and_matrix


__all__ = ['WarpInternode']


TAG_MAPPING = dict(
    image=['image'],
    bbox=['bbox'],
    mask=['mask'],
    point=['point'],
    poly=['poly'],
)


class WarpInternode(DataAugMixin, BaseInternode, BaseFilterMixin):
    def __init__(self, expand=False, ccs=False, tag_mapping=TAG_MAPPING, use_base_filter=True, **kwargs):
        self.expand = expand
        self.ccs = ccs

        self.use_base_filter = use_base_filter

        forward_mapping = dict(
            image=self.forward_image,
            bbox=self.forward_bbox,
            mask=self.forward_mask,
            point=self.forward_point,
            poly=self.forward_poly
        )
        backward_mapping = dict()
        super(WarpInternode, self).__init__(tag_mapping, forward_mapping, backward_mapping, **kwargs)

    def calc_intl_param_forward(self, data_dict):
        raise NotImplementedError

    def forward_rest(self, data_dict, **kwargs):
        if 'intl_warp_rest_matrix' in kwargs.keys():
            data_dict['intl_warp_matrix'] = kwargs['intl_warp_rest_matrix'] @ data_dict['intl_warp_matrix']
        return data_dict

    def forward_image(self, image, meta, intl_warp_tmp_matrix, intl_warp_tmp_size, **kwargs):
        if intl_warp_tmp_matrix is None:
            return image, meta

        M = intl_warp_tmp_matrix
        dst_size = intl_warp_tmp_size

        image = warp_image(image, M, dst_size, self.ccs)
        return image, meta

    def forward_bbox(self, bbox, meta, intl_warp_tmp_matrix, intl_warp_tmp_size, **kwargs):
        if intl_warp_tmp_matrix is None:
            return bbox, meta
        
        M = intl_warp_tmp_matrix
        dst_size = intl_warp_tmp_size

        bbox = warp_bbox(bbox, M)
        bbox = clip_bbox(bbox, dst_size)

        bbox, meta = self.base_filter_bbox(bbox, meta)
        return bbox, meta

    def forward_mask(self, mask, meta, intl_warp_tmp_matrix, intl_warp_tmp_size, **kwargs):
        if intl_warp_tmp_matrix is None:
            return mask, meta

        M = intl_warp_tmp_matrix
        dst_size = intl_warp_tmp_size

        mask = warp_mask(mask, M, dst_size, self.ccs)
        return mask, meta

    def forward_point(self, point, meta, intl_warp_tmp_matrix, intl_warp_tmp_size, **kwargs):
        if intl_warp_tmp_matrix is None:
            return point, meta

        M = intl_warp_tmp_matrix
        dst_size = intl_warp_tmp_size

        n = len(point)
        if n > 0:
            point = point.reshape(-1, 2)
            point = warp_point(point, M)
            point = point.reshape(n, -1, 2)
            point = clip_point(point, dst_size)

        point, meta = self.base_filter_point(point, meta)
        return point, meta

    def forward_poly(self, poly, meta, intl_warp_tmp_matrix, intl_warp_tmp_size, **kwargs):
        if intl_warp_tmp_matrix is None:
            return poly, meta

        M = intl_warp_tmp_matrix
        dst_size = intl_warp_tmp_size

        poly = [warp_point(p, M) for p in poly]
        poly = clip_poly(poly, dst_size)

        poly, meta = self.base_filter_poly(poly, meta)
        return poly, meta

    def __repr__(self):
        return 'expand={}, ccs={}'.format(self.expand, self.ccs)
