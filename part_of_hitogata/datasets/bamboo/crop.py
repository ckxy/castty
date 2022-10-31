import math
import random
import numpy as np
from PIL import Image
from .builder import INTERNODE
from .base_internode import BaseInternode
from .warp_internode import WarpInternode
from ...utils.bbox_tools import calc_iou1, xyxy2xywh
from ..utils.common import get_image_size, is_pil, filter_bbox_by_center, clip_bbox, clip_poly, filter_bbox, filter_point


__all__ = ['Crop', 'AdaptiveCrop', 'AdaptiveTranslate', 'MinIOUCrop', 'MinIOGCrop', 'CenterCrop', 'RandomAreaCrop', 'EastRandomCrop', 'RandomCenterCropPad']


def crop_image(image, x1, y1, x2, y2):
    if is_pil(image):
        image = image.crop((x1, y1, x2, y2))
    else:
        # image = image[y1:y2, x1:x2]
        image = Image.fromarray(image)
        image = image.crop((x1, y1, x2, y2))
        image = np.asarray(image)
    return image


def crop_bbox(bboxes, x1, y1):
    bboxes[:, 0] -= x1
    bboxes[:, 1] -= y1
    bboxes[:, 2] -= x1
    bboxes[:, 3] -= y1
    return bboxes


def crop_poly(polys, x1, y1):
    for i in range(len(polys)):
        polys[i][..., 0] -= x1
        polys[i][..., 1] -= y1
    return polys


def crop_point(points, x1, y1):
    points[..., 0] -= x1
    points[..., 1] -= y1
    return points


def crop_mask(mask, x1, y1, x2, y2):
    mask = mask[y1:y2, x1:x2]
    return mask


@INTERNODE.register_module()
class Crop(BaseInternode):
    def __init__(self, size, **kwargs):
        assert len(size) == 2 and size[0] > 0 and size[1] > 0
        self.size = size

    def calc_cropping(self, data_dict):
        assert 'point' not in data_dict.keys() and 'bbox' not in data_dict.keys() and 'poly' not in data_dict.keys()

        w, h = get_image_size(data_dict['image'])

        xmin = random.randint(0, w - self.size[0])
        ymin = random.randint(0, h - self.size[1])
        xmax = xmin + self.size[0]
        ymax = ymin + self.size[1]
        return xmin, ymin, xmax, ymax

    def calc_intl_param_forward(self, data_dict):
        data_dict['intl_cropping'] = self.calc_cropping(data_dict)
        return data_dict

    def forward(self, data_dict):
        xmin, ymin, xmax, ymax = data_dict['intl_cropping']

        if 'image' in data_dict.keys():
            data_dict['image'] = crop_image(data_dict['image'], xmin, ymin, xmax, ymax)

        if 'bbox' in data_dict.keys():
            data_dict['bbox'] = crop_bbox(data_dict['bbox'], xmin, ymin)

        if 'point' in data_dict.keys():
            data_dict['point'] = crop_point(data_dict['point'], xmin, ymin)

        if 'poly' in data_dict.keys():
            data_dict['poly'] = crop_poly(data_dict['poly'], xmin, ymin)

        if 'mask' in data_dict.keys():
            data_dict['mask'] = crop_mask(data_dict['mask'], xmin, ymin, xmax, ymax)

        data_dict = self.clip_and_filter(data_dict)

        return data_dict

    def clip_and_filter(self, data_dict):
        xmin, ymin, xmax, ymax = data_dict['intl_cropping']
        dst_size = (xmax - xmin, ymax - ymin)

        if 'bbox' in data_dict.keys():
            boxes = data_dict['bbox'].copy()
            boxes = clip_bbox(boxes, dst_size)
            keep = filter_bbox(boxes)
            data_dict['bbox'] = boxes[keep]

            if 'bbox_meta' in data_dict.keys():
                data_dict['bbox_meta'].filter(keep)

        if 'point' in data_dict.keys():
            n = len(data_dict['point'])
            if n > 0:
                points = data_dict['point'].reshape(-1, 2)

                discard = filter_point(points, dst_size)

                visible = data_dict['point_meta']['visible'].reshape(-1)
                visible[discard] = False
                data_dict['point_meta']['visible'] = visible.reshape(n, -1)

                data_dict['point'] = points.reshape(n, -1, 2)

        if 'poly' in data_dict.keys():
            data_dict['poly'], keep = clip_poly(data_dict['poly'], dst_size)

            if 'poly_meta' in data_dict.keys():
                data_dict['poly_meta'].filter(keep)

        return data_dict

    def erase_intl_param_forward(self, data_dict):
        data_dict.pop('intl_cropping')
        return data_dict

    def __repr__(self):
        return 'Crop(size={})'.format(self.size)


@INTERNODE.register_module()
class AdaptiveCrop(Crop):
    def __init__(self, **kwargs):
        pass

    def calc_cropping(self, data_dict):
        assert 'point' in data_dict.keys() or 'bbox' in data_dict.keys() or 'poly' in data_dict.keys()

        w, h = get_image_size(data_dict['image'])

        box = []
        if 'bbox' in data_dict.keys():
            bboxes = data_dict['bbox']
            box.append(np.array([np.min(bboxes[:, 0]), np.min(bboxes[:, 1]), np.max(bboxes[:, 2]), np.max(bboxes[:, 3])]).astype(np.int))

        if 'point' in data_dict.keys():
            if 'point_meta' in data_dict.keys():
                points = data_dict['point'][data_dict['point_meta']['visible']]
                points = points.reshape(-1, 2)
            else:
                points = data_dict['point'].reshape(-1, 2)
            box.append(np.concatenate((np.min(points, axis=0), np.max(points, axis=0))).astype(np.int))

        if 'poly' in data_dict.keys():
            polys = np.array(data_dict['poly']).reshape(-1, 2)
            box.append(np.concatenate((np.min(polys, axis=0), np.max(polys, axis=0))).astype(np.int))

        box = np.array(box)

        xmin = random.randint(0, np.min(box[:, 0]))
        ymin = random.randint(0, np.min(box[:, 1]))
        xmax = random.randint(np.max(box[:, 2]), w)
        ymax = random.randint(np.max(box[:, 3]), h)
        return xmin, ymin, xmax, ymax

    def clip_and_filter(self, data_dict):
        return data_dict

    def __repr__(self):
        return 'AdaptiveCrop()'


@INTERNODE.register_module()
class AdaptiveTranslate(WarpInternode):
    def calc_cropping(self, data_dict):
        assert 'point' in data_dict.keys() or 'bbox' in data_dict.keys() or 'poly' in data_dict.keys()

        w, h = get_image_size(data_dict['image'])

        box = []
        if 'bbox' in data_dict.keys():
            bboxes = data_dict['bbox']
            box.append(np.array([np.min(bboxes[:, 0]), np.min(bboxes[:, 1]), np.max(bboxes[:, 2]), np.max(bboxes[:, 3])]).astype(np.int))

        if 'point' in data_dict.keys():
            if 'point_meta' in data_dict.keys():
                points = data_dict['point'][data_dict['point_meta']['visible']]
                points = points.reshape(-1, 2)
            else:
                points = data_dict['point'].reshape(-1, 2)
            box.append(np.concatenate((np.min(points, axis=0), np.max(points, axis=0))).astype(np.int))

        if 'poly' in data_dict.keys():
            polys = np.array(data_dict['poly']).reshape(-1, 2)
            box.append(np.concatenate((np.min(polys, axis=0), np.max(polys, axis=0))).astype(np.int))

        box = np.array(box)

        tx = random.randint(-np.min(box[:, 0]), (w - np.max(box[:, 2])))
        ty = random.randint(-np.min(box[:, 1]), (h - np.max(box[:, 3])))

        T = np.eye(3)
        T[0, 2] = tx
        T[1, 2] = ty

        return T

    def calc_intl_param_forward(self, data_dict):
        data_dict['intl_warp_tmp_matrix'] = self.calc_cropping(data_dict)
        data_dict['intl_warp_tmp_size'] = get_image_size(data_dict['image'])
        data_dict = super(AdaptiveTranslate, self).calc_intl_param_forward(data_dict)

        return data_dict


@INTERNODE.register_module()
class MinIOUCrop(Crop):
    def __init__(self, threshs, aspect_ratio=2, attempts=50, **kwargs):
        assert aspect_ratio >= 1

        self.threshs = [None] + list(threshs)
        self.aspect_ratio = aspect_ratio
        self.attempts = attempts

    def calc_cropping(self, data_dict):
        assert 'bbox' in data_dict.keys()

        width, height = get_image_size(data_dict['image'])

        while True:
            mode = random.choice(self.threshs)
            if mode is None:
                return 0, 0, width, height

            min_iou = mode
            # min_iou = -1

            for _ in range(self.attempts):
                w = int(random.uniform(0.3 * width, width))
                h = int(random.uniform(0.3 * height, height))
                # w, h = 184, 213

                if h / w < 1.0 / self.aspect_ratio or h / w > self.aspect_ratio:
                    continue

                left = int(random.uniform(0, width - w))
                top = int(random.uniform(0, height - h))
                # left, top = 257, 35

                rect = np.array([left, top, left + w, top + h]).astype(np.int32)

                overlap = calc_iou1(data_dict['bbox'], rect[np.newaxis, ...])

                # print(overlap, w, h, min_iou, left, top)

                if (overlap < min_iou).any():
                    continue

                # print(overlap, min_iou, (overlap < min_iou).any())

                boxes = crop_bbox(data_dict['bbox'].copy(), rect[0], rect[1])
                keep = filter_bbox_by_center(boxes, (w, h))

                if len(keep) == 0:
                    continue

                return rect[0], rect[1], rect[2], rect[3]

    def clip_and_filter(self, data_dict):
        xmin, ymin, xmax, ymax = data_dict['intl_cropping']
        dst_size = (xmax - xmin, ymax - ymin)

        if 'bbox' in data_dict.keys():
            boxes = data_dict['bbox'].copy()
            keep = filter_bbox_by_center(boxes, dst_size)
            boxes = boxes[keep]
            boxes = clip_bbox(boxes, dst_size)
            data_dict['bbox'] = boxes

            if 'bbox_meta' in data_dict.keys():
                data_dict['bbox_meta'].filter(keep)

        return data_dict

    def __repr__(self):
        return 'MinIOUCrop(iou_threshs={}, aspect_ratio={}, attempts={})'.format(self.threshs, self.aspect_ratio, self.attempts)


@INTERNODE.register_module()
class MinIOGCrop(MinIOUCrop):
    def __init__(self, threshs, aspect_ratio=2, attempts=50, **kwargs):
        super(MinIOGCrop, self).__init__(threshs, aspect_ratio, attempts, **kwargs)
        self.ul = (3 / 10 / self.aspect_ratio, 10 * self.aspect_ratio / 3)

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
        if p / self.aspect_ratio <= upper <= p * self.aspect_ratio:
            ps.append(upper)
        if p / self.aspect_ratio <= lower <= p * self.aspect_ratio:
            ps.append(lower)
        if lower <= p * self.aspect_ratio <= upper:
            ps.append(p * self.aspect_ratio)
        if lower <= p / self.aspect_ratio <= upper:
            ps.append(p / self.aspect_ratio)
        return sorted(list(set(ps)))

    def calc_cropping(self, data_dict):
        assert 'bbox' in data_dict.keys()

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
            mode = random.choice(self.threshs)
            if mode is None:
                return 0, 0, width, height

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

                boxes = crop_bbox(data_dict['bbox'].copy(), rect[0], rect[1])
                keep = filter_bbox_by_center(boxes, (w, h))

                if len(keep) == 0:
                    continue

                return rect[0], rect[1], rect[2], rect[3]

    def __repr__(self):
        return 'MinIOGCrop(iog_threshs={}, aspect_ratio={}, attempts={})'.format(self.threshs, self.aspect_ratio, self.attempts)


@INTERNODE.register_module()
class CenterCrop(Crop):
    def calc_cropping(self, data_dict):
        w, h = get_image_size(data_dict['image'])

        x1 = int(round((w - self.size[0]) / 2.))
        y1 = int(round((h - self.size[1]) / 2.))
        assert x1 >= 0 and y1 >= 0

        return x1, y1, x1 + self.size[0], y1 + self.size[1]

    def __repr__(self):
        return 'CenterCrop(size={})'.format(self.size)


@INTERNODE.register_module()
class RandomAreaCrop(Crop):
    def __init__(self, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), attempts=10, **kwargs):
        assert scale[0] < scale[1]
        assert scale[1] <= 1 and scale[0] > 0
        assert ratio[0] <= ratio[1]

        self.scale = scale
        self.ratio = ratio
        self.attempts = attempts

    def calc_cropping(self, data_dict):
        width, height = get_image_size(data_dict['image'])
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
                # return i, j, h, w
                return j, i, j + w, i + h

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
        # return i, j, h, w
        return j, i, j + w, i + h

    def __repr__(self):
        return 'RandomAreaCrop(scale={}, ratio={}, attempts={})'.format(self.scale, self.ratio, self.attempts)

    def rper(self):
        return 'RandomAreaCrop(not available)'


@INTERNODE.register_module()
class EastRandomCrop(Crop):
    def __init__(self, max_tries=50, min_crop_side_ratio=0.1, **kwargs):
        assert 0 < min_crop_side_ratio <= 1
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio

    def calc_cropping(self, data_dict):
        assert 'poly' in data_dict.keys() or 'bbox' in data_dict.keys()

        polys = []

        if 'poly' in data_dict.keys():
            polys.extend(data_dict['poly'])

        if 'bbox' in data_dict.keys():
            for bbox in data_dict['bbox']:
                polys.append(bbox.reshape(2, 2))

        crop_x, crop_y, crop_w, crop_h = self.crop_area(data_dict['image'], polys)
        return crop_x, crop_y, crop_x + crop_w, crop_y + crop_h

    def is_poly_in_rect(self, poly, x, y, w, h):
        poly = np.array(poly)
        if poly[:, 0].min() < x or poly[:, 0].max() > x + w:
            return False
        if poly[:, 1].min() < y or poly[:, 1].max() > y + h:
            return False
        return True

    def is_poly_outside_rect(self, poly, x, y, w, h):
        poly = np.array(poly).reshape(-1, 2)
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False

    def split_regions(self, axis):
        regions = []
        min_axis = 0
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i - 1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        return regions

    def random_select(self, axis, max_size):
        xx = np.random.choice(axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)
        return xmin, xmax

    def region_wise_random_select(self, regions):
        selected_index = list(np.random.choice(len(regions), 2))
        selected_values = []
        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)
        xmin = min(selected_values)
        xmax = max(selected_values)
        return xmin, xmax

    def crop_area(self, img, polys):
        w, h = get_image_size(img)
        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)
        for points in polys:
            # points = np.round(
            #     points, decimals=0).astype(np.int32).reshape(-1, 2)
            points = np.ceil(points).astype(np.int32).reshape(-1, 2)
            min_x = np.min(points[:, 0])
            max_x = np.max(points[:, 0])
            w_array[min_x:max_x] = 1
            min_y = np.min(points[:, 1])
            max_y = np.max(points[:, 1])
            h_array[min_y:max_y] = 1
        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if len(h_axis) == 0 or len(w_axis) == 0:
            return 0, 0, w, h

        h_regions = self.split_regions(h_axis)
        w_regions = self.split_regions(w_axis)

        for i in range(self.max_tries):
            if len(w_regions) > 1:
                xmin, xmax = self.region_wise_random_select(w_regions)
            else:
                xmin, xmax = self.random_select(w_axis, w)
            if len(h_regions) > 1:
                ymin, ymax = self.region_wise_random_select(h_regions)
            else:
                ymin, ymax = self.random_select(h_axis, h)

            # xmin, xmax, ymin, ymax = 1, 73, 16, 374

            if (xmax - xmin < self.min_crop_side_ratio * w
                    or ymax - ymin < self.min_crop_side_ratio * h):
                # area too small
                continue
            num_poly_in_rect = 0
            for poly in polys:
                if not self.is_poly_outside_rect(poly, xmin, ymin, xmax - xmin,
                                                 ymax - ymin):
                    num_poly_in_rect += 1
                    break

            if num_poly_in_rect > 0:
                return xmin, ymin, xmax - xmin, ymax - ymin

        return 0, 0, w, h

    def __repr__(self):
        return 'EastRandomCrop(max_tries={}, min_crop_side_ratio={})'.format(self.max_tries, self.min_crop_side_ratio)


@INTERNODE.register_module()
class WestRandomCrop(Crop):
    def __init__(self, min_crop_side_ratio=0.1, **kwargs):
        assert 0 < min_crop_side_ratio <= 1
        self.min_crop_side_ratio = min_crop_side_ratio

    def calc_cropping(self, data_dict):
        assert 'poly' in data_dict.keys() or 'bbox' in data_dict.keys()

        polys = []

        if 'poly' in data_dict.keys():
            polys.extend(data_dict['poly'])

        if 'bbox' in data_dict.keys():
            for bbox in data_dict['bbox']:
                polys.append(bbox.reshape(2, 2))

        xmin, ymin, xmax, ymax = self.crop_area(data_dict['image'], polys)
        return xmin, ymin, xmax, ymax

    def split_regions(self, axis, max_axis):
        regions = [[0]]
        for i in axis:
            if i == 0:
                continue
            if i == regions[-1][-1] + 1:
                regions[-1].append(i)
            else:
                regions.append([i])

        if max_axis == regions[-1][-1] + 1:
            regions[-1].append(max_axis)
        else:
            regions.append([max_axis])

        regions = [[r[0], r[-1]] for r in regions]
        return regions

    def calc_regions_distances(self, regions):
        dis = np.zeros([len(regions), len(regions)], dtype=np.float32)
        for i in range(len(regions)):
            for j in range(len(regions)):
                if i != j:
                    t = regions[i] + regions[j]
                    dis[i][j] = max(t) - min(t)
        return dis

    def region_wise_random_select(self, regions, dis, length):
        flags = dis >= length
        x, y = np.nonzero(flags)
        i = np.random.choice(np.arange(len(x), dtype=np.int32), size=1)
        i_min = int(min(x[i], y[i]))
        i_max = int(max(x[i], y[i]))

        left_region = regions[i_min]
        right_region = regions[i_max]

        if left_region[1] <= right_region[1] - length:
            minv = random.randint(left_region[0], left_region[1])
        else:
            minv = random.randint(left_region[0], right_region[1] - length)

        if right_region[0] >= left_region[1] + length:
            maxv = random.randint(right_region[0], right_region[1])
        else:
            maxv = random.randint(minv + length, right_region[1])
        return minv, maxv

    def crop_area(self, img, polys):
        w, h = get_image_size(img)
        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)
        for points in polys:
            points = np.ceil(points).astype(np.int32).reshape(-1, 2)

            min_x = np.min(points[:, 0])
            max_x = np.max(points[:, 0])

            min_y = np.min(points[:, 1])
            max_y = np.max(points[:, 1])

            w_array[min_x:max_x] = 1
            h_array[min_y:max_y] = 1

        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if len(h_axis) == 0 or len(w_axis) == 0:
            return 0, 0, w, h

        h_regions = self.split_regions(h_axis, h)
        w_regions = self.split_regions(w_axis, w)

        h_dis = self.calc_regions_distances(h_regions)
        w_dis = self.calc_regions_distances(w_regions)

        ymin, ymax = self.region_wise_random_select(h_regions, h_dis, self.min_crop_side_ratio * h)
        xmin, xmax = self.region_wise_random_select(w_regions, w_dis, self.min_crop_side_ratio * w)

        return xmin, ymin, xmax, ymax

    def __repr__(self):
        return 'WestRandomCrop(min_crop_side_ratio={})'.format(self.min_crop_side_ratio)


@INTERNODE.register_module()
class RandomCenterCropPad(Crop):
    def __init__(self, size, ratios, border=128, **kwargs):
        assert len(size) == 2 and size[0] > 0 and size[1] > 0
        self.size = size
        self.ratios = ratios
        self.border = border

    def _get_border(self, border, size):
        k = 2 * border / size
        i = pow(2, np.ceil(np.log2(np.ceil(k))) + (k == int(k)))
        return border // i

    def calc_cropping(self, data_dict):
        w, h = get_image_size(data_dict['image'])

        if len(data_dict['bbox']) == 0:
            return 0, 0, w, h

        while True:
            scale = random.choice(self.ratios)

            new_h = int(self.size[0] * scale)
            new_w = int(self.size[1] * scale)
            new_h += 1 if new_h % 2 == 1 else 0
            new_w += 1 if new_w % 2 == 1 else 0

            h_border = self._get_border(self.border, h)
            w_border = self._get_border(self.border, w)

            for i in range(50):
                center_x = np.random.randint(low=w_border, high=w - w_border)
                center_y = np.random.randint(low=h_border, high=h - h_border)

                xmin = center_x - new_w // 2
                ymin = center_y - new_h // 2
                xmax = center_x + new_w // 2
                ymax = center_y + new_h // 2

                boxes = data_dict['bbox'].copy()
                boxes = crop_bbox(boxes, xmin, ymin)
                boxes = clip_bbox(boxes, (new_w, new_h))
                keep = filter_bbox(boxes)

                if len(keep) > 0:
                    return xmin, ymin, xmax, ymax

    def __repr__(self):
        return 'RandomCenterCropPad(size={}, ratios={}, border={})'.format(self.size, self.ratios, self.border)
