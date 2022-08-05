import cv2
import math
import random
import numbers
import numpy as np
from .base_internode import BaseInternode
from torchvision.transforms.functional import pad
from ..utils.common import clip_bbox, filter_bbox, is_pil, get_image_size, clip_poly, filter_point
from .builder import INTERNODE


__all__ = ['Padding', 'PaddingBySize', 'PaddingByStride', 'RandomExpand']


CV2_PADDING_MODES = dict(
    constant=cv2.BORDER_CONSTANT, 
    edge=cv2.BORDER_REPLICATE, 
    reflect=cv2.BORDER_REFLECT_101, 
    symmetric=cv2.BORDER_REFLECT
)


def pad_image(image, padding, fill, padding_mode):
    left, right, top, bottom = padding

    if is_pil(image):
        image = pad(image, (left, top, right, bottom), fill, padding_mode)
    else:
        image = cv2.copyMakeBorder(image, top, bottom, left, right, CV2_PADDING_MODES[padding_mode], value=fill)
    return image


def unpad_image(image, padding):
    left, right, top, bottom = padding
    w, h = get_image_size(image)

    if is_pil(image):
        image = image.crop((left, top, w - right, h - bottom))
    else:
        image = image[top:h - bottom, left:w - right]
    # print(get_image_size(image), 'unpad')
    return image


def pad_bbox(bboxes, padding):
    left, right, top, bottom = padding

    bboxes[:, 0] += left
    bboxes[:, 1] += top
    bboxes[:, 2] += left
    bboxes[:, 3] += top

    return bboxes


def unpad_bbox(bboxes, padding):
    left, right, top, bottom = padding

    bboxes[:, 0] -= left
    bboxes[:, 1] -= top
    bboxes[:, 2] -= left
    bboxes[:, 3] -= top

    return bboxes


def pad_poly(polys, padding):
    left, right, top, bottom = padding

    for i in range(len(polys)):
        polys[i][..., 0] += left
        polys[i][..., 1] += top

    return polys


def unpad_poly(polys, padding):
    left, right, top, bottom = padding

    for i in range(len(polys)):
        polys[i][..., 0] -= left
        polys[i][..., 1] -= top

    return polys


def pad_point(points, padding):
    left, right, top, bottom = padding

    points[..., 0] += left
    points[..., 1] += top

    return points


def unpad_point(points, padding):
    left, right, top, bottom = padding

    points[..., 0] -= left
    points[..., 1] -= top

    return points


def pad_mask(mask, padding, padding_mode):
    left, right, top, bottom = padding

    mask = cv2.copyMakeBorder(mask, top, bottom, left, right, CV2_PADDING_MODES[padding_mode], value=0)
    return mask


def unpad_mask(mask, padding):
    left, right, top, bottom = padding
    w, h = get_image_size(mask)

    mask = mask[top:h - bottom, left:w - right]
    return mask


@INTERNODE.register_module()
class Padding(BaseInternode):
    def __init__(self, padding, fill=0, padding_mode='constant', **kwargs):
        assert isinstance(padding, tuple) and len(padding) == 4
        assert isinstance(fill, (numbers.Number, tuple))
        assert padding_mode in CV2_PADDING_MODES.keys()

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, data_dict):
        data_dict['image'] = pad_image(data_dict['image'], self.padding, self.fill, self.padding_mode)

        if 'bbox' in data_dict.keys():
            data_dict['bbox'][:, 0] += self.padding[0]
            data_dict['bbox'][:, 1] += self.padding[1]
            data_dict['bbox'][:, 2] += self.padding[0]
            data_dict['bbox'][:, 3] += self.padding[1]
        return data_dict

    def __repr__(self):
        return 'Padding(padding={}, fill={}, padding_mode={})'.format(self.padding, self.fill, self.padding_mode)


@INTERNODE.register_module()
class PaddingBySize(BaseInternode):
    def __init__(self, size, fill=0, padding_mode='constant', center=False, **kwargs):
        assert isinstance(size, tuple) and len(size) == 2
        assert isinstance(fill, (numbers.Number, tuple))
        assert padding_mode in CV2_PADDING_MODES.keys()

        self.size = size
        self.fill = fill
        self.padding_mode = padding_mode
        self.center = center

    def calc_padding(self, w, h):
        if self.center:
            left = max((self.size[0] - w) // 2, 0)
            right = max(self.size[0] - w, 0) - left
            top = max((self.size[1] - h) // 2, 0)
            bottom = max(self.size[1] - h, 0) - top
        else:
            left = 0
            right = max(self.size[0] - w, 0)
            top = 0
            bottom = max(self.size[1] - h, 0)
        return left, right, top, bottom

    def __call__(self, data_dict):
        w, h = get_image_size(data_dict['image'])

        padding = self.calc_padding(w, h)

        data_dict['image'] = pad_image(data_dict['image'], padding, self.fill, self.padding_mode)

        if 'bbox' in data_dict.keys():
            data_dict['bbox'] = pad_bbox(data_dict['bbox'], padding)

        if 'poly' in data_dict.keys():
            data_dict['poly'] = pad_poly(data_dict['poly'], padding)

        if 'point' in data_dict.keys():
            data_dict['point'] = pad_point(data_dict['point'], padding)

        if 'mask' in data_dict.keys():
            data_dict['mask'] = pad_mask(data_dict['mask'], padding, self.padding_mode)

        return data_dict

    def reverse(self, **kwargs):
        if 'resize_and_padding_reverse_flag' not in kwargs.keys():
            return kwargs

        if 'ori_size' in kwargs.keys():
            h, w = kwargs['ori_size']
            h, w = int(h), int(w)

            padding = self.calc_padding(w, h)
        else:
            return kwargs

        if 'image' in kwargs.keys():
            kwargs['image'] = unpad_image(kwargs['image'], padding)

        if 'mask' in kwargs.keys():
            kwargs['mask'] = unpad_image(kwargs['mask'], padding)

        if 'bbox' in kwargs.keys():
            kwargs['bbox'] = unpad_bbox(kwargs['bbox'], padding)

            boxes = clip_bbox(kwargs['bbox'], (w, h))
            keep = filter_bbox(boxes)
            kwargs['bbox'] = boxes[keep]

            if 'bbox_meta' in kwargs.keys():
                kwargs['bbox_meta'].filter(keep)

        if 'poly' in kwargs.keys():
            kwargs['poly'] = unpad_poly(kwargs['poly'], padding)

            kwargs['poly'], keep = clip_poly(kwargs['poly'], (w, h))
            if 'poly_meta' in kwargs.keys():
                kwargs['poly_meta'].filter(keep)

        if 'point' in kwargs.keys():
            n = len(kwargs['point'])
            points = kwargs['point'].reshape(-1, 2)
            points = unpad_point(points, padding)

            discard = filter_point(points, (w, h))

            if 'point_meta' in kwargs.keys():
                visible = kwargs['point_meta']['visible'].reshape(-1)
                visible[discard] = False
                kwargs['point_meta']['visible'] = visible.reshape(n, -1)
            else:
                points[discard] = -1

            kwargs['point'] = points.reshape(n, -1, 2)

        return kwargs

    def __repr__(self):
        return 'PaddingBySize(size={}, fill={}, padding_mode={}, center={})'.format(self.size, self.fill, self.padding_mode, self.center)


@INTERNODE.register_module()
class PaddingByStride(BaseInternode):
    def __init__(self, stride, fill=0, padding_mode='constant', center=False, **kwargs):
        assert isinstance(stride, int) and stride > 0
        assert isinstance(fill, (numbers.Number, tuple))
        assert padding_mode in CV2_PADDING_MODES.keys()

        self.stride = stride
        self.fill = fill
        self.padding_mode = padding_mode
        self.center = center

    def calc_padding(self, w, h):
        nw = math.ceil(w / self.stride) * self.stride
        nh = math.ceil(h / self.stride) * self.stride

        if self.center:
            left = (nw - w) // 2
            right = nw - w - left
            top = (nh - h) // 2
            bottom = nh - h - top
        else:
            left = 0
            right = nw - w
            top = 0
            bottom = nh - h
        return left, right, top, bottom

    def __call__(self, data_dict):
        w, h = get_image_size(data_dict['image'])

        padding = self.calc_padding(w, h)

        data_dict['image'] = pad_image(data_dict['image'], padding, self.fill, self.padding_mode)

        if 'bbox' in data_dict.keys():
            data_dict['bbox'] = pad_bbox(data_dict['bbox'], padding)

        if 'poly' in data_dict.keys():
            data_dict['poly'] = pad_poly(data_dict['poly'], padding)

        if 'point' in data_dict.keys():
            data_dict['point'] = pad_point(data_dict['point'], padding)

        if 'mask' in data_dict.keys():
            data_dict['mask'] = pad_mask(data_dict['mask'], padding, self.padding_mode)

        return data_dict

    def reverse(self, **kwargs):
        if 'resize_and_padding_reverse_flag' not in kwargs.keys():
            return kwargs

        if 'ori_size' in kwargs.keys():
            h, w = kwargs['ori_size']
            h, w = int(h), int(w)

            padding = self.calc_padding(w, h)
        else:
            return kwargs

        if 'image' in kwargs.keys():
            kwargs['image'] = unpad_image(kwargs['image'], padding)

        if 'mask' in kwargs.keys():
            kwargs['mask'] = unpad_image(kwargs['mask'], padding)

        if 'bbox' in kwargs.keys():
            kwargs['bbox'] = unpad_bbox(kwargs['bbox'], padding)

            boxes = clip_bbox(kwargs['bbox'], (w, h))
            keep = filter_bbox(boxes)
            kwargs['bbox'] = boxes[keep]

            if 'bbox_meta' in kwargs.keys():
                kwargs['bbox_meta'].filter(keep)

        if 'poly' in kwargs.keys():
            kwargs['poly'] = unpad_poly(kwargs['poly'], padding)

            kwargs['poly'], keep = clip_poly(kwargs['poly'], (w, h))
            if 'poly_meta' in kwargs.keys():
                kwargs['poly_meta'].filter(keep)

        if 'point' in kwargs.keys():
            n = len(kwargs['point'])
            points = kwargs['point'].reshape(-1, 2)
            points = unpad_point(points, padding)

            discard = filter_point(points, (w, h))

            if 'point_meta' in kwargs.keys():
                visible = kwargs['point_meta']['visible'].reshape(-1)
                visible[discard] = False
                kwargs['point_meta']['visible'] = visible.reshape(n, -1)
            else:
                points[discard] = -1

            kwargs['point'] = points.reshape(n, -1, 2)

        return kwargs

    def __repr__(self):
        return 'PaddingByStride(stride={}, fill={}, padding_mode={}, center={})'.format(self.stride, self.fill, self.padding_mode, self.center)


@INTERNODE.register_module()
class RandomExpand(BaseInternode):
    def __init__(self, ratio, fill=0, **kwargs):
        assert ratio > 1
        assert isinstance(fill, (numbers.Number, tuple))

        self.ratio = ratio
        self.fill = fill

    def calc_padding(self, w, h):
        r = random.random() * (self.ratio - 1) + 1

        nw, nh = int(w * r), int(h * r)
        # print(w, h, nw, nh, r)
        left = random.randint(0, nw - w)
        right = nw - w - left
        top = random.randint(0, nh - h)
        bottom = nh - h - top
        return left, right, top, bottom

    def __call__(self, data_dict):
        w, h = get_image_size(data_dict['image'])

        padding = self.calc_padding(w, h)

        data_dict['image'] = pad_image(data_dict['image'], padding, self.fill, 'constant')

        if 'bbox' in data_dict.keys():
            data_dict['bbox'] = pad_bbox(data_dict['bbox'], padding)

        if 'poly' in data_dict.keys():
            data_dict['poly'] = pad_poly(data_dict['poly'], padding)

        if 'point' in data_dict.keys():
            data_dict['point'] = pad_point(data_dict['point'], padding)

        if 'mask' in data_dict.keys():
            data_dict['mask'] = pad_mask(data_dict['mask'], padding, 'constant')

        return data_dict

    def __repr__(self):
        return 'RandomExpand(ratio={}, fill={})'.format(self.ratio, self.fill)
