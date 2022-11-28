# Copyright (c) OpenMMLab. All rights reserved.
# copy from mmocr
import cv2
import sys
import torch
import numpy as np
from ..builder import INTERNODE
from ..base_internode import BaseInternode
from ...utils.common import get_image_size, is_pil, clip_poly
from ..crop import Crop
try:
    import pyclipper
    from shapely.geometry import Polygon as plg
except ImportError:
    pass


__all__ = ['PSEEncode', 'PSEMCEncode', 'PSECrop']


def generate_kernel(img_size, polys, shrink_ratio, max_shrink=sys.maxsize, ignore_flags=None):
    w, h = img_size
    text_kernel = np.zeros((h, w), dtype=np.float32)

    if ignore_flags is None:
        ignore_flags = np.array([False] * len(polys))

    for text_ind, poly in enumerate(polys):
        instance = poly.reshape(-1, 2).astype(np.int32)
        area = plg(instance).area
        peri = cv2.arcLength(instance, True)
        distance = min(
            int(area * (1 - shrink_ratio * shrink_ratio) / (peri + 0.001) +
                0.5), max_shrink)
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(instance, pyclipper.JT_ROUND,
                    pyclipper.ET_CLOSEDPOLYGON)
        shrunk = np.array(pco.Execute(-distance))

        # check shrunk == [] or empty ndarray
        if len(shrunk) == 0 or shrunk.size == 0:
            ignore_flags[text_ind] = True
            continue
        try:
            shrunk = np.array(shrunk[0]).reshape(-1, 2)

        except Exception as e:
            ignore_flags[text_ind] = True
            continue

        if not ignore_flags[text_ind]:
            cv2.fillPoly(text_kernel, [shrunk.astype(np.int32)], 1)
    return text_kernel, ignore_flags


def generate_effective_mask(img_size, polys, ignore_flags):
    w, h = img_size
    mask = np.ones((h, w), dtype=np.uint8)

    for poly, tag in zip(polys, ignore_flags):
        if tag:
            instance = poly.reshape(-1, 2).astype(np.int32).reshape(1, -1, 2)
            cv2.fillPoly(mask, instance, 0)
    return mask


@INTERNODE.register_module()
class PSEEncode(BaseInternode):
    def __init__(self, shrink_ratio=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0), max_shrink=20, **kwargs):
        assert len(shrink_ratio) > 0
        assert max_shrink >= 0

        self.shrink_ratio = shrink_ratio
        self.max_shrink = max_shrink

    def forward(self, data_dict):
        w, h = get_image_size(data_dict['image'])

        if 'poly_meta' in data_dict.keys():
            ignore_flags = data_dict['poly_meta']['ignore_flag']
        else:
            ignore_flags = np.array([False] * len(data_dict['poly']))

        kernels = []
        for s in self.shrink_ratio:
            kernel, ignore_flags = generate_kernel((w, h), data_dict['poly'], s, self.max_shrink, ignore_flags)
            kernels.append(kernel[np.newaxis, ...])
            # ignore_flags = np.bitwise_or(ignore_flags, ignore_flags_tmp)
            # print(ignore_flags_tmp, ignore_flags)

        kernels = np.concatenate(kernels)
        data_dict['ocrdet_kernel'] = torch.from_numpy(kernels)

        train_mask = generate_effective_mask((w, h), data_dict['poly'], ignore_flags)
        data_dict['ocrdet_train_mask'] = torch.from_numpy(train_mask)

        if 'poly_meta' in data_dict.keys():
            data_dict['poly_meta']['ignore_flag'] = ignore_flags

        return data_dict

    def __repr__(self):
        return 'PSEEncode(shrink_ratio={}, max_shrink={})'.format(tuple(self.shrink_ratio), self.max_shrink)


@INTERNODE.register_module()
class PSEMCEncode(BaseInternode):
    def __init__(self, num_classes=1, shrink_ratio=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0), max_shrink=20, **kwargs):
        assert len(shrink_ratio) > 0
        assert max_shrink >= 0

        self.num_classes = num_classes
        self.shrink_ratio = shrink_ratio
        self.max_shrink = max_shrink

    def forward(self, data_dict):
        assert 'class_id' in data_dict['poly_meta'].keys()
        labels = data_dict['poly_meta']['class_id']

        w, h = get_image_size(data_dict['image'])

        if 'poly_meta' in data_dict.keys():
            ignore_flags = data_dict['poly_meta']['ignore_flag']
        else:
            ignore_flags = np.array([False] * len(data_dict['poly']))

        kernels = []
        for i in range(self.num_classes):
            keep = np.nonzero(labels == i)[0].tolist()
            tmp_polys = [data_dict['poly'][k] for k in keep]
            tmp_flags = [ignore_flags[k] for k in keep]

            for s in self.shrink_ratio:
                kernel, tmp_flags = generate_kernel((w, h), tmp_polys, s, self.max_shrink, tmp_flags)
                kernels.append(kernel[np.newaxis, ...])

            for j, k in enumerate(keep):
                ignore_flags[k] = tmp_flags[j]

        kernels = np.concatenate(kernels)
        data_dict['ocrdet_kernel'] = torch.from_numpy(kernels)

        train_mask = generate_effective_mask((w, h), data_dict['poly'], ignore_flags)
        data_dict['ocrdet_train_mask'] = torch.from_numpy(train_mask)

        if 'poly_meta' in data_dict.keys():
            data_dict['poly_meta']['ignore_flag'] = ignore_flags

        return data_dict

    def __repr__(self):
        return 'PSEMCEncode(num_classes={}, shrink_ratio={}, max_shrink={})'.format(self.num_classes, tuple(self.shrink_ratio), self.max_shrink)


@INTERNODE.register_module()
class PSECrop(Crop):
    def __init__(self, size, positive_sample_ratio=5.0 / 8.0, **kwargs):
        assert 0 <= positive_sample_ratio <= 1

        self.size = size
        self.positive_sample_ratio = positive_sample_ratio

    def sample_offset(self, img_gt, img_size):
        w, h = img_size
        t_h, t_w = self.size

        # target size is bigger than origin size
        t_h = t_h if t_h < h else h
        t_w = t_w if t_w < w else w
        if torch.max(img_gt) > 0 and np.random.random_sample() < self.positive_sample_ratio:

            # make sure to crop the positive region

            # the minimum top left to crop positive region (h,w)
            tl = torch.min(torch.vstack(torch.where(img_gt > 0)), dim=1)[0] - torch.tensor([t_h, t_w])
            tl[tl < 0] = 0
            # the maximum top left to crop positive region
            br = torch.max(torch.vstack(torch.where(img_gt > 0)), dim=1)[0] - torch.tensor([t_h, t_w])
            br[br < 0] = 0
            # if br is too big so that crop the outside region of img
            br[0] = min(br[0], h - t_h)
            br[1] = min(br[1], w - t_w)

            h = np.random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
            w = np.random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
        else:
            # make sure not to crop outside of img

            h = np.random.randint(0, h - t_h) if h - t_h > 0 else 0
            w = np.random.randint(0, w - t_w) if w - t_w > 0 else 0

        return (h, w)

    def calc_cropping(self, data_dict):
        w, h = get_image_size(data_dict['image'])

        img_gt = data_dict['ocrdet_kernel']
        img_gt = img_gt.sum(dim=0) > 0
        img_gt = img_gt.type(torch.int32)
        # print(torch.unique(img_gt), torch.max(img_gt))
        # print(img_gt, img_gt.shape)

        top, left = self.sample_offset(img_gt, (w, h))
        right = left + self.size[0]
        bottom = top + self.size[1]

        return left, top, right, bottom

    def __repr__(self):
        return 'PSECrop(size={}, positive_sample_ratio={})'.format(self.size, self.positive_sample_ratio)
