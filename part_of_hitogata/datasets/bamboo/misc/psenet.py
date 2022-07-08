# Copyright (c) OpenMMLab. All rights reserved.
# copy from mmocr
from ..base_internode import BaseInternode
from ...utils.common import get_image_size, is_pil, clip_poly
from ..builder import INTERNODE
import numpy as np
import cv2
import torch
import sys

try:
    import pyclipper
    from shapely.geometry import Polygon as plg
except ImportError:
    pass

try:
    from .pse import pse
except ImportError:
    pass


__all__ = ['PSEEncode', 'PSEDecode', 'PSECrop']


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
    mask = np.ones((h, w))

    for poly, tag in zip(polys, ignore_flags):
        if tag:
            instance = poly.reshape(-1, 2).astype(np.int32).reshape(1, -1, 2)
            cv2.fillPoly(mask, instance, 0)
    return mask


@INTERNODE.register_module()
class PSEEncode(BaseInternode):
    def __init__(self, 
        shrink_ratio=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0), 
        max_shrink=20,
        mask_with_largest_kernel=False,
        min_area=16,
        threshold_point=0.7311,
        threshold_area=0.93,
        **kwargs):
        assert len(shrink_ratio) > 0
        assert max_shrink >= 0

        self.shrink_ratio = shrink_ratio
        self.max_shrink = max_shrink

        self.mask_with_largest_kernel = mask_with_largest_kernel
        self.min_area = min_area
        self.threshold_point = threshold_point
        self.threshold_area = threshold_area

    def __call__(self, data_dict):
        w, h = get_image_size(data_dict['image'])

        if 'poly_meta' in data_dict.keys():
            ind = data_dict['poly_meta'].index('ignore_flag')
            ignore_flags = data_dict['poly_meta'].values[ind]
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
            data_dict['poly_meta'].values[ind] = ignore_flags

        return data_dict

    def reverse(self, **kwargs):
        if 'ocrdet_kernel' in kwargs.keys():
            kernels = kwargs['ocrdet_kernel'].numpy()[::-1]

            masks = kernels >= self.threshold_point
            if self.mask_with_largest_kernel:
                masks[1:] *= masks[:1]
            masks = masks.astype(np.uint8)

            agg_kernel = pse(masks, self.min_area)

            polys = []
            for n in np.unique(agg_kernel):
                mask = agg_kernel == n
                mean_score = np.mean(kernels[0][mask])
                # print(n, type(n), np.sum(mask), mean_score)

                if mean_score < self.threshold_area:
                    continue

                points = np.array(np.where(mask)[::-1]).transpose((1, 0))

                if len(points) < self.min_area:
                    continue

                rect = cv2.minAreaRect(points)
                bbox = cv2.boxPoints(rect)
                polys.append(bbox)

            kwargs['poly'] = polys
        return kwargs

    def __repr__(self):
        return 'PSEEncode(shrink_ratio={}, max_shrink={})'.format(tuple(self.shrink_ratio), self.max_shrink)

    def rper(self):
        return 'PSEDecode(mask_with_largest_kernel={}, min_area={}, threshold_point={}, threshold_area={})'.format(self.mask_with_largest_kernel, self.min_area, self.threshold_point, self.threshold_area)


@INTERNODE.register_module()
class PSEDecode(PSEEncode):
    def __init__(self, 
        mask_with_largest_kernel=False,
        min_area=16,
        threshold_point=0.7311,
        threshold_area=0.93,
        **kwargs):
        super(PSEDecode, self).__init__(
            mask_with_largest_kernel=mask_with_largest_kernel,
            min_area=min_area,
            threshold_point=threshold_point,
            threshold_area=threshold_area,
            **kwargs
        )

    def __call__(self, data_dict):
        return self.reverse(data_dict)

    def __repr__(self):
        return self.rper()


@INTERNODE.register_module()
class PSECrop(BaseInternode):
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

    def __call__(self, data_dict):
        w, h = get_image_size(data_dict['image'])

        img_gt = data_dict['ocrdet_kernel']
        img_gt = img_gt.sum(dim=0) > 0
        img_gt = img_gt.type(torch.int32)
        # print(torch.unique(img_gt), torch.max(img_gt))
        # print(img_gt, img_gt.shape)

        top, left = self.sample_offset(img_gt, (w, h))
        right = left + self.size[0]
        bottom = top + self.size[1]

        # print(left, right, w, top, bottom, h)

        if is_pil(data_dict['image']):
            data_dict['image'] = data_dict['image'].crop((left, top, right, bottom))
        else:
            data_dict['image'] = data_dict['image'][top:bottom, left:right]

        data_dict['ocrdet_kernel'] = data_dict['ocrdet_kernel'][:, top:bottom, left:right]
        data_dict['ocrdet_train_mask'] = data_dict['ocrdet_train_mask'][top:bottom, left:right]

        if 'poly' in data_dict.keys():
            for i in range(len(data_dict['poly'])):
                data_dict['poly'][i][..., 0] -= left
                data_dict['poly'][i][..., 1] -= top

            data_dict['poly'], keep = clip_poly(data_dict['poly'], (w, h))
            if 'poly_meta' in data_dict.keys():
                data_dict['poly_meta'].filter(keep)

        return data_dict

    def __repr__(self):
        return 'PSECrop(size={}, positive_sample_ratio={})'.format(self.size, self.positive_sample_ratio)
