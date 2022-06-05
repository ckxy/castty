import math
import random
import numpy as np
from PIL import Image
from utils.bbox_tools import calc_iou1, xyxy2xywh
from .base_internode import BaseInternode
from .builder import INTERNODE
from ..utils.common import get_image_size, is_pil, filter_bbox_by_center, clip_bbox
from .warp_internode import WarpInternode


__all__ = ['Crop', 'AdaptiveCrop', 'AdaptiveTranslate', 'MinIOUCrop', 'MinIOGCrop', 'CenterCrop', 'RandomAreaCrop']


@INTERNODE.register_module()
class Crop(BaseInternode):
    def __init__(self, size, **kwargs):
        assert len(size) == 2 and size[0] > 0 and size[1] > 0
        self.size = size

    def __call__(self, data_dict):
        assert 'point' not in data_dict.keys() and 'bbox' not in data_dict.keys() and 'quad' not in data_dict.keys()
        if random.random() < self.p:
            w, h = data_dict['image'].size

            xmin = random.randint(0, w - self.size[0])
            ymin = random.randint(0, h - self.size[1])
            xmax = xmin + self.size[0]
            ymax = ymin + self.size[1]

            data_dict['image'] = data_dict['image'].crop((xmin, ymin, xmax, ymax))

            if 'mask' in data_dict.keys():
                data_dict['mask'] = data_dict['mask'].crop((xmin, ymin, xmax, ymax))

        return data_dict

    def __repr__(self):
        return 'Crop(size={})'.format(self.size)


@INTERNODE.register_module()
class AdaptiveCrop(BaseInternode):
    def __call__(self, data_dict):
        assert 'point' in data_dict.keys() or 'bbox' in data_dict.keys() or 'quad' in data_dict.keys()

        # w, h = data_dict['image'].size
        w, h = get_image_size(data_dict['image'])

        box = []
        if 'bbox' in data_dict.keys():
            bboxes = data_dict['bbox'][:, :4]
            box.append(np.array([np.min(bboxes[:, 0]), np.min(bboxes[:, 1]), np.max(bboxes[:, 2]), np.max(bboxes[:, 3])]).astype(np.int))

        # if 'point' in data_dict.keys():
        #     box.append(np.concatenate((np.min(data_dict['point'], axis=0), np.max(data_dict['point'], axis=0))).astype(np.int))

        box = np.array(box)

        xmin = random.randint(0, np.min(box[:, 0]))
        ymin = random.randint(0, np.min(box[:, 1]))
        xmax = random.randint(np.max(box[:, 2]), w)
        ymax = random.randint(np.max(box[:, 3]), h)

        # xmin, ymin, xmax, ymax = 3, 186, 381, 374
        # print(xmin, ymin, xmax, ymax)

        if 'bbox' in data_dict.keys():
            data_dict['bbox'][:, 0] -= xmin
            data_dict['bbox'][:, 1] -= ymin
            data_dict['bbox'][:, 2] -= xmin
            data_dict['bbox'][:, 3] -= ymin

        # if 'point' in data_dict.keys():
        #     data_dict['point'][:, 0] -= xmin
        #     data_dict['point'][:, 1] -= ymin

        if is_pil(data_dict['image']):
            data_dict['image'] = data_dict['image'].crop((xmin, ymin, xmax, ymax))
        else:
            data_dict['image'] = data_dict['image'][ymin:ymax, xmin:xmax]

        # if 'mask' in data_dict.keys():
        #     data_dict['mask'] = data_dict['mask'].crop((xmin, ymin, xmax, ymax))

        return data_dict


@INTERNODE.register_module()
class AdaptiveTranslate(WarpInternode):
    def __call__(self, data_dict):
        assert 'point' in data_dict.keys() or 'bbox' in data_dict.keys() or 'quad' in data_dict.keys()

        w, h = get_image_size(data_dict['image'])

        box = []
        if 'bbox' in data_dict.keys():
            bboxes = data_dict['bbox'][:, :4]
            box.append(np.array([np.min(bboxes[:, 0]), np.min(bboxes[:, 1]), np.max(bboxes[:, 2]), np.max(bboxes[:, 3])]).astype(np.int))

        # if 'point' in data_dict.keys():
        #     box.append(np.concatenate((np.min(data_dict['point'], axis=0), np.max(data_dict['point'], axis=0))).astype(np.int))

        box = np.array(box)

        tx = random.randint(-np.min(box[:, 0]), (w - np.max(box[:, 2])))
        ty = random.randint(-np.min(box[:, 1]), (h - np.max(box[:, 3])))

        T = np.eye(3)
        T[0, 2] = tx
        T[1, 2] = ty

        data_dict['warp_tmp_matrix'] = T
        data_dict['warp_tmp_size'] = get_image_size(data_dict['image'])
        data_dict = super(AdaptiveTranslate, self).__call__(data_dict)

        return data_dict


@INTERNODE.register_module()
class MinIOUCrop(BaseInternode):
    def __init__(self, threshs, aspect_ratio=2, attempts=50, **kwargs):
        assert aspect_ratio >= 1

        self.iou_threshs = [None] + list(threshs)
        self.aspect_ratio = aspect_ratio
        self.attempts = attempts

    def __call__(self, data_dict):
        assert 'bbox' in data_dict.keys()

        width, height = get_image_size(data_dict['image'])
        while True:
            mode = random.choice(self.iou_threshs)
            if mode is None:
                return data_dict

            min_iou = mode
            # min_iou = -1

            for _ in range(self.attempts):
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # w, h = 184.90499498805883, 296.81326760646834

                if h / w < 1.0 / self.aspect_ratio or h / w > self.aspect_ratio:
                    continue

                left = random.uniform(0, width - w)
                top = random.uniform(0, height - h)
                # left, top = 42.06855911088178, 25.003141919689657
                rect = np.array([left, top, left + w, top + h]).astype(np.int)
                # print(data_dict['bbox'][:, :-1].shape, rect[np.newaxis, ...].shape)
                # print(data_dict['bbox'][:, :-1])
                # print(rect[np.newaxis, ...])
                # exit()

                overlap = calc_iou1(data_dict['bbox'], rect[np.newaxis, ...])

                # print(overlap, w, h, min_iou, left, top)

                if (overlap < min_iou).any():
                    continue
                # print(overlap, min_iou, (overlap < min_iou).any())

                data_dict['bbox'][..., 0] -= rect[0]
                data_dict['bbox'][..., 1] -= rect[1]
                data_dict['bbox'][..., 2] -= rect[0]
                data_dict['bbox'][..., 3] -= rect[1]

                keep = filter_bbox_by_center(data_dict['bbox'], (w, h))

                if len(keep) == 0:
                    continue

                data_dict['bbox'] = data_dict['bbox'][keep]
                data_dict['bbox'] = clip_bbox(data_dict['bbox'], (w, h))
                if 'bbox_meta' in data_dict.keys():
                    data_dict['bbox_meta'].filter(keep)

                if is_pil(data_dict['image']):
                    data_dict['image'] = data_dict['image'].crop((rect[0], rect[1], rect[2], rect[3]))
                else:
                    data_dict['image'] = data_dict['image'][rect[1]:rect[3], rect[0]:rect[2]]

                # centers = (data_dict['bbox'][:, :2] + data_dict['bbox'][:, 2:4]) / 2.0
                # print(centers)
                # print(rect)
                # m1 = (rect[0] <= centers[:, 0]) * (rect[1] <= centers[:, 1])
                # m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                # print(m1, m2, centers[2, 0] >= rect[0], centers[2, 1] >= rect[1])
                # mask = m1 * m2
                # print(mask)

                # if not mask.any():
                #     continue

                # current_image = data_dict['image'].crop((rect[0], rect[1], rect[2], rect[3]))
                # current_boxes = data_dict['bbox'][mask, :].copy()

                # current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                # current_boxes[:, :2] -= rect[:2]

                # current_boxes[:, 2:4] = np.minimum(current_boxes[:, 2:4], rect[2:])
                # current_boxes[:, 2:4] -= rect[:2]

                # data_dict['image'] = current_image
                # data_dict['bbox'] = current_boxes

                return data_dict

    def __repr__(self):
        return 'MinIOUCrop(iou_threshs={}, aspect_ratio={}, attempts={})'.format(self.iou_threshs, self.aspect_ratio, self.attempts)


@INTERNODE.register_module()
class MinIOGCrop(BaseInternode):
    def __init__(self, threshs, aspect_ratio=2, attempts=50, **kwargs):
        assert aspect_ratio >= 1

        self.iog_threshs = [None] + list(threshs)
        self.ar = aspect_ratio
        self.attempts = attempts
        self.ul = (3 / 10 / self.ar, 10 * self.ar / 3)

    def iog_calc(self, boxes1, boxes2):
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        # union_area = boxes1_area + boxes2_area - inter_area
        return inter_area / boxes1_area

    def check(self, p, lower, upper):
        ps = []
        if p / self.ar <= upper <= p * self.ar:
            ps.append(upper)
        if p / self.ar <= lower <= p * self.ar:
            ps.append(lower)
        if lower <= p * self.ar <= upper:
            ps.append(p * self.ar)
        if lower <= p / self.ar <= upper:
            ps.append(p / self.ar)
        return sorted(list(set(ps)))

    def __call__(self, data_dict):
        ulw, ulh = get_image_size(data_dict['image'])
        # assert self.ul[0] * ulh <= ulw <= self.ul[1] * ulh
        if not (self.ul[0] * ulh <= ulw <= self.ul[1] * ulh):
            return data_dict
        llw, llh = 0.3 * ulw, 0.3 * ulh

        # llw
        p_llw = [[llw, p] for p in self.check(llw, llh, ulh)]
        # ulh
        p_ulh = [[p, ulh] for p in self.check(ulh, llw, ulw)]
        # ulw
        p_ulw = [[ulw, p] for p in self.check(ulw, llh, ulh)[::-1]]
        # llh
        p_llh = [[p, llh] for p in self.check(llh, llw, ulw)[::-1]]
        
        temp = p_llw + p_ulh + p_ulw + p_llh
        ps = [temp[0]]
        for i in range(1, len(temp)):
            if temp[i] != ps[-1]:
                ps.append(temp[i])
        if ps[0] == ps[-1]:
            ps = ps[:-1]
        ps = np.array(ps)
        x = ps[:, 0]
        y = ps[:, 1]

        while True:
            mode = random.choice(self.iog_threshs)
            if mode is None:
                return data_dict

            min_iou = mode

            r = np.random.rand(1, len(ps))
            r = r * r
            rs = np.broadcast_to(np.sum(r, axis=1, keepdims=True), r.shape)
            r = r / rs
            xp = np.broadcast_to(x[np.newaxis, ...], r.shape) * r
            yp = np.broadcast_to(y[np.newaxis, ...], r.shape) * r
            w = np.sum(xp, axis=1).astype(np.int)[0]
            h = np.sum(yp, axis=1).astype(np.int)[0]

            for _ in range(self.attempts):
                left = random.uniform(0, ulw - w)
                top = random.uniform(0, ulh - h)
                rect = np.array([left, top, left + w, top + h]).astype(np.int)

                overlap = self.iog_calc(data_dict['bbox'][:, :4], rect[np.newaxis, ...])

                if (overlap < min_iou).any():
                    continue

                data_dict['bbox'][..., 0] -= rect[0]
                data_dict['bbox'][..., 1] -= rect[1]
                data_dict['bbox'][..., 2] -= rect[0]
                data_dict['bbox'][..., 3] -= rect[1]

                keep = filter_bbox_by_center(data_dict['bbox'], (w, h))

                if len(keep) == 0:
                    continue

                data_dict['bbox'] = data_dict['bbox'][keep]
                data_dict['bbox'] = clip_bbox(data_dict['bbox'], (w, h))
                if 'bbox_meta' in data_dict.keys():
                    data_dict['bbox_meta'].filter(keep)

                if is_pil(data_dict['image']):
                    data_dict['image'] = data_dict['image'].crop((rect[0], rect[1], rect[2], rect[3]))
                else:
                    data_dict['image'] = data_dict['image'][rect[1]:rect[3], rect[0]:rect[2]]

                # centers = (data_dict['bbox'][:, :2] + data_dict['bbox'][:, 2:4]) / 2.0
                # m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                # m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                # mask = m1 * m2

                # if not mask.any():
                #     continue

                # current_image = data_dict['image'].crop((rect[0], rect[1], rect[2], rect[3]))
                # current_boxes = data_dict['bbox'][mask, :].copy()

                # current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                # current_boxes[:, :2] -= rect[:2]

                # current_boxes[:, 2:4] = np.minimum(current_boxes[:, 2:4], rect[2:])
                # current_boxes[:, 2:4] -= rect[:2]

                # data_dict['image'] = current_image
                # data_dict['bbox'] = current_boxes

                return data_dict

    def __repr__(self):
        return 'MinIOGCrop(iog_threshs={}, aspect_ratio={}, attempts={})'.format(self.iog_threshs, self.ar, self.attempts)


class CenterCrop(BaseInternode):
    def __init__(self, size):
        assert len(size) == 2 and size[0] > 0 and size[1] > 0
        self.size = size

    def __call__(self, data_dict):
        assert 'point' not in data_dict.keys() and 'bbox' not in data_dict.keys()
        w, h = data_dict['image'].size

        x1 = int(round((w - self.size[0]) / 2.))
        y1 = int(round((h - self.size[1]) / 2.))
        assert x1 >= 0 and y1 >= 0

        data_dict['image'] = data_dict['image'].crop((x1, y1, x1 + self.size[0], y1 + self.size[1]))

        if 'mask' in data_dict.keys():
            data_dict['mask'] = data_dict['mask'].crop((x1, y1, x1 + self.size[0], y1 + self.size[1]))

        return data_dict

    def __repr__(self):
        return 'CenterCrop(size={})'.format(self.size)

    def rper(self):
        return 'CenterCrop(not available)'


class RandomAreaCrop(BaseInternode):
    def __init__(self, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), attempts=10):
        assert scale[0] < scale[1]
        assert scale[1] <= 1 and scale[0] > 0
        assert ratio[0] <= ratio[1]

        self.scale = scale
        self.ratio = ratio
        self.attempts = attempts

    def _calc_crop_coor(self, height, width):
        area = height * width

        for attempt in range(self.attempts):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(self.ratio)):
            w = width
            h = int(round(w / min(self.ratio)))
        elif (in_ratio > max(self.ratio)):
            h = height
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, data_dict):
        assert 'point' not in data_dict.keys() and 'bbox' not in data_dict.keys()

        w, h = data_dict['image'].size
        top, left, height, width = self._calc_crop_coor(h, w)

        data_dict['image'] = data_dict['image'].crop((left, top, left + width, top + height))

        if 'mask' in data_dict.keys():
            data_dict['mask'] = data_dict['mask'].crop((left, top, left + width, top + height))

        return data_dict

    def __repr__(self):
        return 'RandomAreaCrop(scale={}, ratio={}, attempts={})'.format(self.scale, self.ratio, self.attempts)

    def rper(self):
        return 'RandomAreaCrop(not available)'
