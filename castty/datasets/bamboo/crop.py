import math
import random
import numpy as np
from PIL import Image
from .builder import INTERNODE
from .base_internode import BaseInternode
from .warp_internode import WarpInternode
from .mixin import BaseFilterMixin, DataAugMixin
from ...utils.bbox_tools import calc_iou1, xyxy2xywh
from torchvision.transforms.functional import crop as tensor_crop
from ..utils.common import get_image_size, is_pil, is_cv2, filter_bbox_by_center, filter_bbox_by_length, clip_bbox, clip_point, clip_poly


__all__ = ['Crop', 'AdaptiveCrop', 'AdaptiveTranslate', 'MinIOUCrop', 'MinIOGCrop', 'CenterCrop', 'RandomAreaCrop', 'EastRandomCrop', 'WestRandomCrop', 'RandomCenterCropPad']


def crop_image(image, x1, y1, x2, y2):
    if is_pil(image):
        image = image.crop((x1, y1, x2, y2))
    elif is_cv2(image):
        # image = image[y1:y2, x1:x2]
        image = Image.fromarray(image)
        image = image.crop((x1, y1, x2, y2))
        image = np.array(image)
    else:
        image = tensor_crop(image, y1, x1, y2 - y1, x2 - x1)
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
    if is_cv2(mask):
        mask = Image.fromarray(mask)
        mask = mask.crop((x1, y1, x2, y2))
        mask = np.array(mask)
    else:
        mask = mask.unsqueeze(0)
        mask = tensor_crop(mask, y1, x1, y2 - y1, x2 - x1)
        mask = mask[0]
    return mask


TAG_MAPPING = dict(
    image=['image'],
    bbox=['bbox'],
    mask=['mask'],
    point=['point'],
    poly=['poly'],
)


class CropInternode(DataAugMixin, BaseInternode, BaseFilterMixin):
    def __init__(self, tag_mapping=TAG_MAPPING, use_base_filter=True, **kwargs):
        forward_mapping = dict(
            image=self.forward_image,
            bbox=self.forward_bbox,
            mask=self.forward_mask,
            point=self.forward_point,
            poly=self.forward_poly
        )
        backward_mapping = dict()
        # super(CropInternode, self).__init__(tag_mapping, forward_mapping, backward_mapping, **kwargs)
        DataAugMixin.__init__(self, tag_mapping, forward_mapping, backward_mapping)
        BaseInternode.__init__(self, **kwargs)
        BaseFilterMixin.__init__(self, use_base_filter)

    def calc_cropping(self, data_dict):
        raise NotImplementedError

    def calc_intl_param_forward(self, data_dict):
        xmin, ymin, xmax, ymax = self.calc_cropping(data_dict)
        return dict(intl_cropping=(xmin, ymin, xmax, ymax))

    def forward_image(self, image, meta, intl_cropping, **kwargs):
        xmin, ymin, xmax, ymax = intl_cropping
        image = crop_image(image, xmin, ymin, xmax, ymax)
        return image, meta

    def forward_bbox(self, bbox, meta, intl_cropping, **kwargs):        
        xmin, ymin, xmax, ymax = intl_cropping
        dst_size = (xmax - xmin, ymax - ymin)

        bbox = crop_bbox(bbox, xmin, ymin)
        bbox = clip_bbox(bbox, dst_size)

        bbox, meta = self.base_filter_bbox(bbox, meta)
        return bbox, meta

    def forward_mask(self, mask, meta, intl_cropping, **kwargs):
        xmin, ymin, xmax, ymax = intl_cropping
        mask = crop_mask(mask, xmin, ymin, xmax, ymax)
        return mask, meta

    def forward_point(self, point, meta, intl_cropping, **kwargs):        
        xmin, ymin, xmax, ymax = intl_cropping
        dst_size = (xmax - xmin, ymax - ymin)

        point = crop_point(point, xmin, ymin)
        point = clip_point(point, dst_size)

        point, meta = self.base_filter_point(point, meta)
        return point, meta

    def forward_poly(self, poly, meta, intl_cropping, **kwargs):        
        xmin, ymin, xmax, ymax = intl_cropping
        dst_size = (xmax - xmin, ymax - ymin)

        poly = crop_poly(poly, xmin, ymin)
        poly = clip_poly(poly, dst_size)

        poly, meta = self.base_filter_poly(poly, meta)
        return poly, meta


@INTERNODE.register_module()
class Crop(CropInternode):
    def __init__(self, size, tag_mapping=TAG_MAPPING, use_base_filter=True, **kwargs):
        assert len(size) == 2 and size[0] > 0 and size[1] > 0
        self.size = size

        # super(Crop, self).__init__(tag_mapping=tag_mapping, use_base_filter=use_base_filter, **kwargs)
        CropInternode.__init__(self, tag_mapping=tag_mapping, use_base_filter=use_base_filter, **kwargs)

    def calc_cropping(self, data_dict):
        assert 'point' not in data_dict.keys() and 'bbox' not in data_dict.keys() and 'poly' not in data_dict.keys()

        w, h = get_image_size(data_dict['image'])

        xmin = random.randint(0, w - self.size[0])
        ymin = random.randint(0, h - self.size[1])
        xmax = xmin + self.size[0]
        ymax = ymin + self.size[1]
        xmin, ymin, xmax, ymax = 0, 200, 600, 600
        return xmin, ymin, xmax, ymax

    def __repr__(self):
        return 'CropBySize(size={})'.format(self.size)


@INTERNODE.register_module()
class AdaptiveCrop(CropInternode):
    def calc_cropping(self, data_dict):
        assert 'point' in data_dict.keys() or 'bbox' in data_dict.keys() or 'poly' in data_dict.keys()

        w, h = get_image_size(data_dict['image'])

        box = []
        if 'bbox' in data_dict.keys():
            bboxes = data_dict['bbox']
            box.append(np.array([np.min(bboxes[:, 0]), np.min(bboxes[:, 1]), np.max(bboxes[:, 2]), np.max(bboxes[:, 3])]).astype(np.int32))

        if 'point' in data_dict.keys():
            points = data_dict['point'][data_dict['point_meta']['keep']]
            points = points.reshape(-1, 2)
            box.append(np.concatenate((np.min(points, axis=0), np.max(points, axis=0))).astype(np.int32))

        if 'poly' in data_dict.keys():
            polys = np.array(data_dict['poly']).reshape(-1, 2)
            box.append(np.concatenate((np.min(polys, axis=0), np.max(polys, axis=0))).astype(np.int32))

        box = np.array(box)

        xmin = random.randint(0, np.min(box[:, 0]))
        ymin = random.randint(0, np.min(box[:, 1]))
        xmax = random.randint(np.max(box[:, 2]), w)
        ymax = random.randint(np.max(box[:, 3]), h)
        return xmin, ymin, xmax, ymax

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
            box.append(np.array([np.min(bboxes[:, 0]), np.min(bboxes[:, 1]), np.max(bboxes[:, 2]), np.max(bboxes[:, 3])]).astype(np.int32))

        if 'point' in data_dict.keys():
            points = data_dict['point'][data_dict['point_meta']['keep']]
            points = points.reshape(-1, 2)
            box.append(np.concatenate((np.min(points, axis=0), np.max(points, axis=0))).astype(np.int32))

        if 'poly' in data_dict.keys():
            polys = np.array(data_dict['poly']).reshape(-1, 2)
            box.append(np.concatenate((np.min(polys, axis=0), np.max(polys, axis=0))).astype(np.int32))

        box = np.array(box)

        tx = random.randint(-np.min(box[:, 0]), (w - np.max(box[:, 2])))
        ty = random.randint(-np.min(box[:, 1]), (h - np.max(box[:, 3])))

        T = np.eye(3)
        T[0, 2] = tx
        T[1, 2] = ty

        return T

    def calc_intl_param_forward(self, data_dict):
        # data_dict['intl_warp_tmp_matrix'] = self.calc_cropping(data_dict)
        # data_dict['intl_warp_tmp_size'] = get_image_size(data_dict['image'])
        # data_dict = super(AdaptiveTranslate, self).calc_intl_param_forward(data_dict)

        return dict(intl_warp_tmp_matrix=self.calc_cropping(data_dict), intl_warp_tmp_size=get_image_size(data_dict['image']))


@INTERNODE.register_module()
class MinIOUCrop(CropInternode):
    def __init__(self, threshs, aspect_ratio=2, attempts=50, tag_mapping=TAG_MAPPING, use_base_filter=True, **kwargs):
        assert aspect_ratio >= 1

        self.threshs = [None] + list(threshs)
        self.aspect_ratio = aspect_ratio
        self.attempts = attempts

        # super(MinIOUCrop, self).__init__(tag_mapping=tag_mapping, use_base_filter=use_base_filter, **kwargs)
        CropInternode.__init__(self, tag_mapping=tag_mapping, use_base_filter=use_base_filter, **kwargs)

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

                # rect = [363, 171, 745, 872]
                # print(rect)

                return rect[0], rect[1], rect[2], rect[3]

    def __repr__(self):
        return 'MinIOUCrop(iou_threshs={}, aspect_ratio={}, attempts={})'.format(self.threshs, self.aspect_ratio, self.attempts)


@INTERNODE.register_module()
class MinIOGCrop(MinIOUCrop):
    def __init__(self, threshs, aspect_ratio=2, attempts=50, tag_mapping=TAG_MAPPING, use_base_filter=True, **kwargs):
        # super(MinIOGCrop, self).__init__(threshs, aspect_ratio, attempts, tag_mapping, use_base_filter, **kwargs)
        MinIOUCrop.__init__(self, threshs, aspect_ratio, attempts, tag_mapping, use_base_filter, **kwargs)
        self.ul = (3 / 10 / self.aspect_ratio, 10 * self.aspect_ratio / 3)

    @staticmethod
    def iog_calc(boxes1, boxes2):
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
                return 0, 0, ulw, ulh

            min_iou = mode

            r = np.random.rand(1, len(ps))
            r = r * r
            rs = np.broadcast_to(np.sum(r, axis=1, keepdims=True), r.shape)
            r = r / rs
            xp = np.broadcast_to(x[np.newaxis, ...], r.shape) * r
            yp = np.broadcast_to(y[np.newaxis, ...], r.shape) * r
            w = np.sum(xp, axis=1).astype(np.int32)[0]
            h = np.sum(yp, axis=1).astype(np.int32)[0]

            for _ in range(self.attempts):
                left = random.uniform(0, ulw - w)
                top = random.uniform(0, ulh - h)
                rect = np.array([left, top, left + w, top + h]).astype(np.int32)

                overlap = self.iog_calc(data_dict['bbox'][:, :4], rect[np.newaxis, ...])

                if (overlap < min_iou).any():
                    continue

                boxes = crop_bbox(data_dict['bbox'].copy(), rect[0], rect[1])
                keep = filter_bbox_by_center(boxes, (w, h))

                if len(keep) == 0:
                    continue

                # rect = [363, 171, 745, 872]
                # print(rect)

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
class RandomAreaCrop(CropInternode):
    def __init__(self, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), attempts=10, tag_mapping=TAG_MAPPING, use_base_filter=True, **kwargs):
        assert scale[0] < scale[1]
        assert scale[1] <= 1 and scale[0] > 0
        assert ratio[0] <= ratio[1]

        self.scale = scale
        self.ratio = ratio
        self.attempts = attempts

        # super(RandomAreaCrop, self).__init__(tag_mapping=tag_mapping, use_base_filter=use_base_filter, **kwargs)
        CropInternode.__init__(self, tag_mapping=tag_mapping, use_base_filter=use_base_filter, **kwargs)

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


@INTERNODE.register_module()
class EastRandomCrop(CropInternode):
    def __init__(self, max_tries=50, min_crop_side_ratio=0.1, tag_mapping=TAG_MAPPING, use_base_filter=True, **kwargs):
        assert 0 < min_crop_side_ratio <= 1
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio

        # super(EastRandomCrop, self).__init__(tag_mapping=tag_mapping, use_base_filter=use_base_filter, **kwargs)
        CropInternode.__init__(self, tag_mapping=tag_mapping, use_base_filter=use_base_filter, **kwargs)

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
class WestRandomCrop(CropInternode):
    def __init__(self, min_crop_side_ratio=0.1, tag_mapping=TAG_MAPPING, use_base_filter=True, **kwargs):
        assert 0 < min_crop_side_ratio <= 1
        self.min_crop_side_ratio = min_crop_side_ratio

        # super(WestRandomCrop, self).__init__(tag_mapping=tag_mapping, use_base_filter=use_base_filter, **kwargs)
        CropInternode.__init__(self, tag_mapping=tag_mapping, use_base_filter=use_base_filter, **kwargs)

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

        ymin, ymax = self.region_wise_random_select(h_regions, h_dis, int(self.min_crop_side_ratio * h))
        xmin, xmax = self.region_wise_random_select(w_regions, w_dis, int(self.min_crop_side_ratio * w))

        return xmin, ymin, xmax, ymax

    def __repr__(self):
        return 'WestRandomCrop(min_crop_side_ratio={})'.format(self.min_crop_side_ratio)


@INTERNODE.register_module()
class RandomCenterCropPad(CropInternode):
    def __init__(self, size, ratios, border=128, tag_mapping=TAG_MAPPING, use_base_filter=True, **kwargs):
        assert len(size) == 2 and size[0] > 0 and size[1] > 0
        self.size = size
        self.ratios = ratios
        self.border = border

        # super(RandomCenterCropPad, self).__init__(tag_mapping=tag_mapping, use_base_filter=use_base_filter, **kwargs)
        CropInternode.__init__(self, tag_mapping=tag_mapping, use_base_filter=use_base_filter, **kwargs)

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
                keep = filter_bbox_by_length(boxes)

                if len(keep) > 0:
                    # print(xmin, ymin, xmax, ymax)
                    return xmin, ymin, xmax, ymax

    def __repr__(self):
        return 'RandomCenterCropPad(size={}, ratios={}, border={})'.format(self.size, self.ratios, self.border)
