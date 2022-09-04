import cv2
import math
import random
import numbers
import numpy as np
from .builder import INTERNODE
from .base_internode import BaseInternode
from torchvision.transforms.functional import pad
from .crop import crop_image, crop_bbox, crop_poly, crop_point, crop_mask
from ..utils.common import clip_bbox, filter_bbox, is_pil, get_image_size, clip_poly, filter_point


__all__ = ['Padding', 'PaddingBySize', 'PaddingByStride', 'RandomExpand']


CV2_PADDING_MODES = dict(
    constant=cv2.BORDER_CONSTANT, 
    edge=cv2.BORDER_REPLICATE, 
    reflect=cv2.BORDER_REFLECT_101, 
    symmetric=cv2.BORDER_REFLECT
)


def pad_image(image, padding, fill=(0, 0, 0), padding_mode='constant'):
    left, top, right, bottom = padding

    if is_pil(image):
        image = pad(image, (left, top, right, bottom), fill, padding_mode)
    else:
        image = cv2.copyMakeBorder(image, top, bottom, left, right, CV2_PADDING_MODES[padding_mode], value=fill)
    return image


def pad_bbox(bboxes, left, top):
    bboxes[:, 0] += left
    bboxes[:, 1] += top
    bboxes[:, 2] += left
    bboxes[:, 3] += top
    return bboxes


def pad_poly(polys, left, top):
    for i in range(len(polys)):
        polys[i][..., 0] += left
        polys[i][..., 1] += top
    return polys


def pad_point(points, left, top):
    points[..., 0] += left
    points[..., 1] += top
    return points


def pad_mask(mask, padding, padding_mode):
    left, top, right, bottom = padding
    mask = cv2.copyMakeBorder(mask, top, bottom, left, right, CV2_PADDING_MODES[padding_mode], value=0)
    return mask


def unpad_image(image, padding):
    left, top, right, bottom = padding
    w, h = get_image_size(image)
    return crop_image(image, left, top, w - right, h - bottom)


def unpad_bbox(bboxes, left, top):
    return crop_bbox(bboxes, left, top)


def unpad_poly(polys, left, top):
    return crop_poly(polys, left, top)


def unpad_point(points, left, top):
    return crop_point(points, left, top)


def unpad_mask(mask, padding):
    left, top, right, bottom = padding
    w, h = get_image_size(mask)
    return crop_mask(mask, left, top, w - right, h - bottom)


@INTERNODE.register_module()
class Padding(BaseInternode):
    def __init__(self, padding, fill=(0, 0, 0), padding_mode='constant', **kwargs):
        assert isinstance(padding, tuple) and len(padding) == 4
        assert isinstance(fill, tuple) and len(fill) == 3
        assert padding_mode in CV2_PADDING_MODES.keys()

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def calc_padding(self, w, h):
        return self.padding[0], self.padding[1], self.padding[2], self.padding[3]

    def calc_intl_param_forward(self, data_dict):
        w, h = get_image_size(data_dict['image'])
        data_dict['intl_padding'] = self.calc_padding(w, h)
        return data_dict

    def forward(self, data_dict):
        left, top, right, bottom = data_dict['intl_padding']

        if 'image' in data_dict.keys():
            data_dict['image'] = pad_image(data_dict['image'], (left, top, right, bottom), self.fill, self.padding_mode)

        if 'bbox' in data_dict.keys():
            data_dict['bbox'] = pad_bbox(data_dict['bbox'], left, top)

        if 'poly' in data_dict.keys():
            data_dict['poly'] = pad_poly(data_dict['poly'], left, top)

        if 'point' in data_dict.keys():
            data_dict['point'] = pad_point(data_dict['point'], left, top)

        if 'mask' in data_dict.keys():
            data_dict['mask'] = pad_mask(data_dict['mask'], (left, top, right, bottom), self.padding_mode)
        return data_dict

    def erase_intl_param_forward(self, data_dict):
        data_dict.pop('intl_padding')
        return data_dict

    def __call__(self, data_dict):
        data_dict = self.calc_intl_param_forward(data_dict)
        data_dict = self.forward(data_dict)
        data_dict = self.erase_intl_param_forward(data_dict)
        return data_dict

    def __repr__(self):
        return 'Padding(padding={}, fill={}, padding_mode={})'.format(self.padding, self.fill, self.padding_mode)


@INTERNODE.register_module()
class PaddingBySize(Padding):
    def __init__(self, size, fill=(0, 0, 0), padding_mode='constant', center=False, **kwargs):
        assert isinstance(size, tuple) and len(size) == 2
        assert isinstance(fill, tuple) and len(fill) == 3
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
        return left, top, right, bottom

    def calc_intl_param_backward(self, data_dict):
        if 'intl_resize_and_padding_reverse_flag' in data_dict.keys():
            h, w = data_dict['ori_size']
            h, w = int(h), int(w)

            data_dict['intl_padding'] = self.calc_padding(w, h)
        return data_dict

    def backward(self, data_dict):
        if 'intl_padding' in data_dict.keys():
            left, top, right, bottom = data_dict['intl_padding']
            h, w = data_dict['ori_size']
            h, w = int(h), int(w)

            if 'image' in data_dict.keys():
                data_dict['image'] = unpad_image(data_dict['image'], (left, top, right, bottom))

            if 'mask' in data_dict.keys():
                data_dict['mask'] = unpad_mask(data_dict['mask'], (left, top, right, bottom))

            if 'bbox' in data_dict.keys():
                data_dict['bbox'] = unpad_bbox(data_dict['bbox'], left, top)

                boxes = clip_bbox(data_dict['bbox'], (w, h))
                keep = filter_bbox(boxes)
                data_dict['bbox'] = boxes[keep]

                if 'bbox_meta' in data_dict.keys():
                    data_dict['bbox_meta'].filter(keep)

            if 'poly' in data_dict.keys():
                data_dict['poly'] = unpad_poly(data_dict['poly'], left, top)

                data_dict['poly'], keep = clip_poly(data_dict['poly'], (w, h))
                if 'poly_meta' in data_dict.keys():
                    data_dict['poly_meta'].filter(keep)

            if 'point' in data_dict.keys():
                n = len(data_dict['point'])
                points = data_dict['point'].reshape(-1, 2)
                points = unpad_point(points, left, top)

                discard = filter_point(points, (w, h))

                if 'point_meta' in data_dict.keys():
                    visible = data_dict['point_meta']['visible'].reshape(-1)
                    visible[discard] = False
                    data_dict['point_meta']['visible'] = visible.reshape(n, -1)
                else:
                    points[discard] = -1

                data_dict['point'] = points.reshape(n, -1, 2)
        return data_dict

    def erase_intl_param_backward(self, data_dict):
        if 'intl_padding' in data_dict.keys():
            data_dict = self.erase_intl_param_forward(data_dict)
        return data_dict

    def reverse(self, **kwargs):
        # if 'intl_resize_and_padding_reverse_flag' not in kwargs.keys():
        #     return kwargs

        # if 'ori_size' in kwargs.keys():
        #     h, w = kwargs['ori_size']
        #     h, w = int(h), int(w)

        #     left, top, right, bottom = self.calc_padding(w, h)
        # else:
        #     return kwargs

        # if 'image' in kwargs.keys():
        #     kwargs['image'] = unpad_image(kwargs['image'], (left, top, right, bottom))

        # if 'mask' in kwargs.keys():
        #     kwargs['mask'] = unpad_mask(kwargs['mask'], (left, top, right, bottom))

        # if 'bbox' in kwargs.keys():
        #     kwargs['bbox'] = unpad_bbox(kwargs['bbox'], left, top)

        #     boxes = clip_bbox(kwargs['bbox'], (w, h))
        #     keep = filter_bbox(boxes)
        #     kwargs['bbox'] = boxes[keep]

        #     if 'bbox_meta' in kwargs.keys():
        #         kwargs['bbox_meta'].filter(keep)

        # if 'poly' in kwargs.keys():
        #     kwargs['poly'] = unpad_poly(kwargs['poly'], left, top)

        #     kwargs['poly'], keep = clip_poly(kwargs['poly'], (w, h))
        #     if 'poly_meta' in kwargs.keys():
        #         kwargs['poly_meta'].filter(keep)

        # if 'point' in kwargs.keys():
        #     n = len(kwargs['point'])
        #     points = kwargs['point'].reshape(-1, 2)
        #     points = unpad_point(points, left, top)

        #     discard = filter_point(points, (w, h))

        #     if 'point_meta' in kwargs.keys():
        #         visible = kwargs['point_meta']['visible'].reshape(-1)
        #         visible[discard] = False
        #         kwargs['point_meta']['visible'] = visible.reshape(n, -1)
        #     else:
        #         points[discard] = -1

        #     kwargs['point'] = points.reshape(n, -1, 2)
        kwargs = self.calc_intl_param_backward(kwargs)
        kwargs = self.backward(kwargs)
        kwargs = self.erase_intl_param_backward(kwargs)
        return kwargs

    def __repr__(self):
        return 'PaddingBySize(size={}, fill={}, padding_mode={}, center={})'.format(self.size, self.fill, self.padding_mode, self.center)


@INTERNODE.register_module()
class PaddingByStride(PaddingBySize):
    def __init__(self, stride, fill=(0, 0, 0), padding_mode='constant', center=False, **kwargs):
        assert isinstance(stride, int) and stride > 0
        assert isinstance(fill, tuple) and len(fill) == 3
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
        return left, top, right, bottom

    def __repr__(self):
        return 'PaddingByStride(stride={}, fill={}, padding_mode={}, center={})'.format(self.stride, self.fill, self.padding_mode, self.center)


@INTERNODE.register_module()
class RandomExpand(Padding):
    def __init__(self, ratio, fill=0, **kwargs):
        assert ratio > 1
        assert isinstance(fill, tuple) and len(fill) == 3

        self.ratio = ratio
        self.fill = fill

    def calc_padding(self, w, h):
        r = random.random() * (self.ratio - 1) + 1

        nw, nh = int(w * r), int(h * r)
        left = random.randint(0, nw - w)
        right = nw - w - left
        top = random.randint(0, nh - h)
        bottom = nh - h - top
        return left, top, right, bottom

    def __repr__(self):
        return 'RandomExpand(ratio={}, fill={})'.format(self.ratio, self.fill)
