import os
import cv2
import random
from PIL import Image
from .builder import INTERNODE
from .mixin import DataAugMixin
from .base_internode import BaseInternode
from ..utils.common import get_image_size, is_pil


__all__ = ['Flip']


TAG_MAPPING = dict(
    image=['image'],
    bbox=['bbox'],
    mask=['mask'],
    point=['point'],
    poly=['poly'],
)


@INTERNODE.register_module()
class Flip(DataAugMixin, BaseInternode):
    def __init__(self, horizontal=True, mapping=None, tag_mapping=TAG_MAPPING, **kwargs):
        self.horizontal = horizontal

        if mapping is None:
            self.map_idx = None
        else:
            with open(os.path.join(mapping), 'r') as f:
                lines = f.readlines()
            assert len(lines) == 1
            map_idx = lines[0].strip().split(',')
            self.map_idx = list(map(int, map_idx))
            self.map_path = mapping
            
        forward_mapping = dict(
            image=self.forward_image,
            bbox=self.forward_bbox,
            mask=self.forward_mask,
            point=self.forward_point,
            poly=self.forward_poly
        )
        backward_mapping = dict()
        super(Flip, self).__init__(tag_mapping, forward_mapping, backward_mapping, **kwargs)

    def calc_intl_param_forward(self, data_dict):
        return dict(intl_flip_wh=get_image_size(data_dict['image']))

    def forward_image(self, image, meta, intl_flip_wh, **kwargs):
        if is_pil(image):
            mode = Image.FLIP_LEFT_RIGHT if self.horizontal else Image.FLIP_TOP_BOTTOM
            image = image.transpose(mode)
        else:
            mode = 1 if self.horizontal else 0
            image = cv2.flip(image, mode)
        return image, meta

    def forward_bbox(self, bbox, meta, intl_flip_wh, **kwargs):
        w, h = intl_flip_wh
        
        if self.horizontal:
            bbox[:, 0], bbox[:, 2] = w - bbox[:, 2], w - bbox[:, 0]
        else:
            bbox[:, 1], bbox[:, 3] = h - bbox[:, 3], h - bbox[:, 1]

        return bbox, meta

    def forward_mask(self, mask, meta, intl_flip_wh, **kwargs):        
        mode = 1 if self.horizontal else 0
        mask = cv2.flip(mask, mode)
        return mask, meta

    def forward_point(self, point, meta, intl_flip_wh, **kwargs):
        w, h = intl_flip_wh
        
        if self.horizontal:
            point[..., 0] = w - point[..., 0]
        else:
            point[..., 1] = h - point[..., 1]
        if self.map_idx is not None:
            for i in range(len(point)):
                point[i] = point[i, self.map_idx]

        return point, meta

    def forward_poly(self, poly, meta, intl_flip_wh, **kwargs):
        w, h = intl_flip_wh
        
        if self.horizontal:
            for i in range(len(poly)):
                poly[i][:, 0] = w - poly[i][:, 0]
        else:
            for i in range(len(poly)):
                poly[i][:, 1] = h - poly[i][:, 1]

        return poly, meta

    def __repr__(self):
        if self.map_idx is None:
            return 'Flip(horizontal={})'.format(self.horizontal)
        else:
            return 'Flip(horizontal={}, map={})'.format(self.horizontal, self.map_path)
