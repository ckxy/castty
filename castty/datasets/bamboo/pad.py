import cv2
import math
import random
import numbers
import numpy as np
from .builder import INTERNODE
from .base_internode import BaseInternode
from .mixin import BaseFilterMixin, DataAugMixin
from torchvision.transforms.functional import pad as tensor_pad
from .crop import crop_image, crop_bbox, crop_poly, crop_point, crop_mask
from ..utils.common import is_pil, is_cv2, get_image_size, clip_bbox, clip_point, clip_poly

import numbers
from PIL import Image, ImageOps
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union


__all__ = ['Padding', 'PaddingBySize', 'PaddingByStride', 'RandomExpand']


TAG_MAPPING = dict(
    image=['image'],
    bbox=['bbox'],
    mask=['mask'],
    point=['point'],
    poly=['poly'],
)


CV2_PADDING_MODES = dict(
    constant=cv2.BORDER_CONSTANT, 
    edge=cv2.BORDER_REPLICATE, 
    reflect=cv2.BORDER_REFLECT_101, 
    symmetric=cv2.BORDER_REFLECT
)


# copy from torchvision/transforms/_functional_pil.py
def get_image_num_channels(img: Any) -> int:
    if is_pil(img):
        if hasattr(img, "getbands"):
            return len(img.getbands())
        else:
            return img.channels
    raise TypeError(f"Unexpected type {type(img)}")


# copy from torchvision/transforms/_functional_pil.py
def _parse_fill(
    fill: Optional[Union[float, List[float], Tuple[float, ...]]],
    img: Image.Image,
    name: str = "fillcolor",
) -> Dict[str, Optional[Union[float, List[float], Tuple[float, ...]]]]:

    # Process fill color for affine transforms
    num_channels = get_image_num_channels(img)
    if fill is None:
        fill = 0
    if isinstance(fill, (int, float)) and num_channels > 1:
        fill = tuple([fill] * num_channels)
    if isinstance(fill, (list, tuple)):
        if len(fill) != num_channels:
            msg = "The number of elements in 'fill' does not match the number of channels of the image ({} != {})"
            raise ValueError(msg.format(len(fill), num_channels))

        fill = tuple(fill)

    if img.mode != "F":
        if isinstance(fill, (list, tuple)):
            fill = tuple(int(x) for x in fill)
        else:
            fill = int(fill)

    return {name: fill}


# copy from torchvision/transforms/_functional_pil.py
def pad(
    img: Image.Image,
    padding: Union[int, List[int], Tuple[int, ...]],
    fill: Optional[Union[float, List[float], Tuple[float, ...]]] = 0,
    padding_mode: Literal["constant", "edge", "reflect", "symmetric"] = "constant",
) -> Image.Image:

    if not is_pil(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    if not isinstance(padding, (numbers.Number, tuple, list)):
        raise TypeError("Got inappropriate padding arg")
    if fill is not None and not isinstance(fill, (numbers.Number, tuple, list)):
        raise TypeError("Got inappropriate fill arg")
    if not isinstance(padding_mode, str):
        raise TypeError("Got inappropriate padding_mode arg")

    if isinstance(padding, list):
        padding = tuple(padding)

    if isinstance(padding, tuple) and len(padding) not in [1, 2, 4]:
        raise ValueError(f"Padding must be an int or a 1, 2, or 4 element tuple, not a {len(padding)} element tuple")

    if isinstance(padding, tuple) and len(padding) == 1:
        # Compatibility with `functional_tensor.pad`
        padding = padding[0]

    if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
        raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")

    if padding_mode == "constant":
        opts = _parse_fill(fill, img, name="fill")
        if img.mode == "P":
            palette = img.getpalette()
            image = ImageOps.expand(img, border=padding, **opts)
            image.putpalette(palette)
            return image

        return ImageOps.expand(img, border=padding, **opts)
    else:
        if isinstance(padding, int):
            pad_left = pad_right = pad_top = pad_bottom = padding
        if isinstance(padding, tuple) and len(padding) == 2:
            pad_left = pad_right = padding[0]
            pad_top = pad_bottom = padding[1]
        if isinstance(padding, tuple) and len(padding) == 4:
            pad_left = padding[0]
            pad_top = padding[1]
            pad_right = padding[2]
            pad_bottom = padding[3]

        p = [pad_left, pad_top, pad_right, pad_bottom]
        cropping = -np.minimum(p, 0)

        if cropping.any():
            crop_left, crop_top, crop_right, crop_bottom = cropping
            img = img.crop((crop_left, crop_top, img.width - crop_right, img.height - crop_bottom))

        pad_left, pad_top, pad_right, pad_bottom = np.maximum(p, 0)

        if img.mode == "P":
            palette = img.getpalette()
            img = np.asarray(img)
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode=padding_mode)
            img = Image.fromarray(img)
            img.putpalette(palette)
            return img

        img = np.asarray(img)
        # RGB image
        if len(img.shape) == 3:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), padding_mode)
        # Grayscale image
        if len(img.shape) == 2:
            img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), padding_mode)

        return Image.fromarray(img)


def pad_image(image, padding, fill=(0, 0, 0), padding_mode='constant'):
    left, top, right, bottom = padding

    if is_pil(image):
        image = pad(image, (left, top, right, bottom), fill, padding_mode)
    elif is_cv2(image):
        image = cv2.copyMakeBorder(image, top, bottom, left, right, CV2_PADDING_MODES[padding_mode], value=fill)
    else:
        image = tensor_pad(image, (left, top, right, bottom), None, padding_mode)
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
    if is_cv2(mask):
        mask = cv2.copyMakeBorder(mask, top, bottom, left, right, CV2_PADDING_MODES[padding_mode], value=0)
    else:
        mask = pad(mask, (left, top, right, bottom), None, padding_mode)
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


class PaddingInternode(DataAugMixin, BaseInternode):
    def __init__(self, fill=(0, 0, 0), padding_mode='constant', tag_mapping=TAG_MAPPING, **kwargs):
        assert isinstance(fill, tuple) and len(fill) == 3
        assert padding_mode in CV2_PADDING_MODES.keys()

        self.fill = fill
        self.padding_mode = padding_mode

        forward_mapping = dict(
            image=self.forward_image,
            bbox=self.forward_bbox,
            mask=self.forward_mask,
            point=self.forward_point,
            poly=self.forward_poly
        )
        backward_mapping = dict()
        # super(PaddingInternode, self).__init__(tag_mapping, forward_mapping, backward_mapping, **kwargs)
        DataAugMixin.__init__(self, tag_mapping, forward_mapping, backward_mapping)
        BaseInternode.__init__(self, **kwargs)

    def calc_padding(self, w, h):
        raise NotImplementedError

    def calc_intl_param_forward(self, data_dict):
        w, h = get_image_size(data_dict['image'])
        return dict(intl_padding=self.calc_padding(w, h))

    def forward_image(self, image, meta, intl_padding, **kwargs):
        left, top, right, bottom = intl_padding
        image = pad_image(image, (left, top, right, bottom), self.fill, self.padding_mode)
        return image, meta

    def forward_bbox(self, bbox, meta, intl_padding, **kwargs):
        left, top, right, bottom = intl_padding
        bbox = pad_bbox(bbox, left, top)
        return bbox, meta

    def forward_mask(self, mask, meta, intl_padding, **kwargs):
        left, top, right, bottom = intl_padding
        mask = pad_mask(mask, (left, top, right, bottom), self.padding_mode)
        return mask, meta

    def forward_point(self, point, meta, intl_padding, **kwarg):
        left, top, right, bottom = intl_padding
        point = pad_point(point, left, top)
        return point, meta

    def forward_poly(self, poly, meta, intl_padding, **kwarg):
        left, top, right, bottom = intl_padding
        poly = pad_poly(poly, left, top)
        return poly, meta


@INTERNODE.register_module()
class Padding(PaddingInternode):
    def __init__(self, padding, fill=(0, 0, 0), padding_mode='constant', tag_mapping=TAG_MAPPING, **kwargs):
        assert isinstance(padding, tuple) and len(padding) == 4
        self.padding = padding

        # super(Padding, self).__init__(fill=fill, padding_mode=padding_mode, tag_mapping=tag_mapping, **kwargs)
        PaddingInternode.__init__(self, fill=fill, padding_mode=padding_mode, tag_mapping=tag_mapping, **kwargs)

    def calc_padding(self, w, h):
        return self.padding[0], self.padding[1], self.padding[2], self.padding[3]

    def __repr__(self):
        return 'Padding(padding={}, fill={}, padding_mode={})'.format(self.padding, self.fill, self.padding_mode)


class ReversiblePadding(PaddingInternode, BaseFilterMixin):
    def __init__(self, fill=(0, 0, 0), padding_mode='constant', tag_mapping=TAG_MAPPING, use_base_filter=True, **kwargs):
        # self.use_base_filter = use_base_filter
        BaseFilterMixin.__init__(self, use_base_filter)

        # super(ReversiblePadding, self).__init__(fill=fill, padding_mode=padding_mode, tag_mapping=tag_mapping, **kwargs)
        PaddingInternode.__init__(self, fill=fill, padding_mode=padding_mode, tag_mapping=tag_mapping, **kwargs)
        self.backward_mapping = dict(
            image=self.backward_image,
            bbox=self.backward_bbox,
            mask=self.backward_mask,
            point=self.backward_point,
            poly=self.backward_poly
        )

    def calc_intl_param_backward(self, data_dict):
        if 'intl_resize_and_padding_reverse_flag' in data_dict.keys():
            w, h = data_dict['ori_size']
            return dict(intl_padding=self.calc_padding(w, h), intl_ori_size=data_dict['ori_size'])
        else:
            return dict(intl_padding=None, intl_ori_size=None)

    def backward_image(self, image, meta, intl_padding, intl_ori_size, **kwargs):
        if intl_padding is not None:
            left, top, right, bottom = intl_padding
        else:
            return image, meta

        image = unpad_image(image, (left, top, right, bottom))
        return image, meta

    def backward_bbox(self, bbox, meta, intl_padding, intl_ori_size, **kwargs):
        if intl_padding is not None:
            left, top, right, bottom = intl_padding
            w, h = intl_ori_size
        else:
            return bbox, meta

        bbox = unpad_bbox(bbox, left, top)
        bbox = clip_bbox(bbox, (w, h))

        bbox, meta = self.base_filter_bbox(bbox, meta)
        return bbox, meta

    def backward_mask(self, mask, meta, intl_padding, intl_ori_size, **kwargs):
        if intl_padding is not None:
            left, top, right, bottom = intl_padding
        else:
            return mask, meta

        mask = unpad_mask(mask, (left, top, right, bottom))
        return mask, meta

    def backward_point(self, point, meta, intl_padding, intl_ori_size, **kwarg):
        if intl_padding is not None:
            left, top, right, bottom = intl_padding
            w, h = intl_ori_size
        else:
            return point, meta

        point = unpad_point(point, left, top)
        point = clip_point(point, (w, h))

        point, meta = self.base_filter_point(point, meta)
        return point, meta

    def backward_poly(self, poly, meta, intl_padding, intl_ori_size, **kwarg):
        if intl_padding is not None:
            left, top, right, bottom = intl_padding
            w, h = intl_ori_size
        else:
            return poly, meta

        poly = unpad_poly(poly, left, top)
        poly = clip_poly(poly, (w, h))

        poly, meta = self.base_filter_poly(poly, meta)
        return poly, meta

    def __repr__(self):
        return 'PaddingBySize(size={}, fill={}, padding_mode={}, center={})'.format(self.size, self.fill, self.padding_mode, self.center)


@INTERNODE.register_module()
class PaddingBySize(ReversiblePadding):
    def __init__(self, size, center=False, fill=(0, 0, 0), padding_mode='constant', tag_mapping=TAG_MAPPING, use_base_filter=True, **kwargs):
        assert isinstance(size, tuple) and len(size) == 2

        self.size = size
        self.center = center

        # super(PaddingBySize, self).__init__(fill=fill, padding_mode=padding_mode, tag_mapping=tag_mapping, use_base_filter=True, **kwargs)
        ReversiblePadding.__init__(self, fill=fill, padding_mode=padding_mode, tag_mapping=tag_mapping, use_base_filter=use_base_filter, **kwargs)

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

    def __repr__(self):
        return 'PaddingBySize(size={}, fill={}, padding_mode={}, center={})'.format(self.size, self.fill, self.padding_mode, self.center)


@INTERNODE.register_module()
class PaddingByStride(ReversiblePadding):
    def __init__(self, stride, center=False, fill=(0, 0, 0), padding_mode='constant', tag_mapping=TAG_MAPPING, use_base_filter=True, **kwargs):
        assert isinstance(stride, int) and stride > 0

        self.stride = stride
        self.center = center

        # super(PaddingByStride, self).__init__(fill=fill, padding_mode=padding_mode, use_base_filter=True, **kwargs)
        ReversiblePadding.__init__(self, fill=fill, padding_mode=padding_mode, tag_mapping=tag_mapping, use_base_filter=use_base_filter, **kwargs)

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
class RandomExpand(PaddingInternode):
    def __init__(self, ratio, fill=(0, 0, 0), padding_mode='constant', tag_mapping=TAG_MAPPING, **kwargs):
        assert ratio > 1
        self.ratio = ratio

        # super(RandomExpand, self).__init__(fill=fill, padding_mode=padding_mode, tag_mapping=tag_mapping, **kwargs)
        PaddingInternode.__init__(self, fill=fill, padding_mode=padding_mode, tag_mapping=tag_mapping, **kwargs)

    def calc_padding(self, w, h):
        r = random.random() * (self.ratio - 1) + 1

        nw, nh = int(w * r), int(h * r)
        left = random.randint(0, nw - w)
        right = nw - w - left
        top = random.randint(0, nh - h)
        bottom = nh - h - top
        return left, top, right, bottom

    def __repr__(self):
        return 'RandomExpand(ratio={}, fill={}, padding_mode={})'.format(self.ratio, self.fill, self.padding_mode)
