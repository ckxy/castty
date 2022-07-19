# Copyright (c) OpenMMLab. All rights reserved.
# copy from mmocr
from ..base_internode import BaseInternode
from ...utils.common import get_image_size, is_pil, clip_poly
from ..builder import INTERNODE
import numpy as np
import cv2
import torch
from .psenet import generate_effective_mask, generate_kernel

try:
    import pyclipper
    from shapely.geometry import Polygon
except ImportError:
    pass


__all__ = ['DBEncode', 'DBMCEncode']


def polygon_area(polygon):
    polygon = polygon.reshape(-1, 2)
    edge = 0
    for i in range(polygon.shape[0]):
        next_index = (i + 1) % polygon.shape[0]
        edge += (polygon[next_index, 0] - polygon[i, 0]) * (
            polygon[next_index, 1] + polygon[i, 1])

    return edge / 2.


def polygon_size(polygon):
    poly = polygon.reshape(-1, 2)
    rect = cv2.minAreaRect(poly.astype(np.int32))
    size = rect[1]
    return size


def point2line(xs, ys, point_1, point_2):
    # a^2
    a_square = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
    # b^2
    b_square = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
    # c^2
    c_square = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] -
                                                              point_2[1])
    # -cosC=(c^2-a^2-b^2)/2(ab)
    neg_cos_c = (
        (c_square - a_square - b_square) /
        (np.finfo(np.float32).eps + 2 * np.sqrt(a_square * b_square)))

    # python浮点数问题会导致neg_cos_c的绝对值超过1，即使类型都为float64也不行
    neg_cos_c = np.clip(neg_cos_c, -1, 1)

    # sinC^2=1-cosC^2
    square_sin = 1 - np.square(neg_cos_c)
    square_sin = np.nan_to_num(square_sin)
    # distance=a*b*sinC/c=a*h/c=2*area/c
    result = np.sqrt(a_square * b_square * square_sin /
                     (np.finfo(np.float32).eps + c_square))
    # set result to minimum edge if C<pi/2
    result[neg_cos_c < 0] = np.sqrt(np.fmin(a_square,
                                            b_square))[neg_cos_c < 0]
    return result


@INTERNODE.register_module()
class DBEncode(BaseInternode):
    def __init__(self, 
        shrink_ratio=0.4,
        thr_min=0.3,
        thr_max=0.7,
        min_short_size=8,
        **kwargs):
        self.shrink_ratio = shrink_ratio
        self.thr_min = thr_min
        self.thr_max = thr_max
        self.min_short_size = min_short_size

    def find_invalid(self, polys, ignore_flags):
        for idx, poly in enumerate(polys):
            if not ignore_flags[idx] and self.invalid_polygon(poly):
                ignore_flags[idx] = True
        return ignore_flags

    def invalid_polygon(self, poly):
        area = polygon_area(poly)
        if abs(area) < 1:
            return True
        short_size = min(polygon_size(poly))
        if short_size < self.min_short_size:
            return True

        return False

    def generate_thr_map(self, img_size, polygons, ignore_flags):
        thr_map = np.zeros(img_size, dtype=np.float32)
        thr_mask = np.zeros(img_size, dtype=np.uint8)

        for i, polygon in enumerate(polygons):
            if ignore_flags[i]:
                continue
            self.draw_border_map(polygon.copy(), thr_map, mask=thr_mask)
        thr_map = thr_map * (self.thr_max - self.thr_min) + self.thr_min

        return thr_map, thr_mask

    def draw_border_map(self, polygon, canvas, mask):
        polygon = polygon.reshape(-1, 2)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)
        distance = (
            polygon_shape.area * (1 - np.power(self.shrink_ratio, 2)) /
            polygon_shape.length)
        subject = [tuple(p) for p in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = padding.Execute(distance)
        if len(padded_polygon) > 0:
            padded_polygon = np.array(padded_polygon[0])
        else:
            print(f'padding {polygon} with {distance} gets {padded_polygon}')
            padded_polygon = polygon.copy().astype(np.int32)

        x_min = padded_polygon[:, 0].min()
        x_max = padded_polygon[:, 0].max()
        y_min = padded_polygon[:, 1].min()
        y_max = padded_polygon[:, 1].max()

        width = x_max - x_min + 1
        height = y_max - y_min + 1

        polygon[:, 0] = polygon[:, 0] - x_min
        polygon[:, 1] = polygon[:, 1] - y_min

        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width),
            (height, width)).astype(np.float32)
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1),
            (height, width)).astype(np.float32)

        distance_map = np.zeros((polygon.shape[0], height, width),
                                dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = point2line(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        x_min_valid = min(max(0, x_min), canvas.shape[1] - 1)
        x_max_valid = min(max(0, x_max), canvas.shape[1] - 1)
        y_min_valid = min(max(0, y_min), canvas.shape[0] - 1)
        y_max_valid = min(max(0, y_max), canvas.shape[0] - 1)

        if x_min_valid - x_min >= width or y_min_valid - y_min >= height:
            return

        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)
        canvas[y_min_valid:y_max_valid + 1,
               x_min_valid:x_max_valid + 1] = np.fmax(
                   1 - distance_map[y_min_valid - y_min:y_max_valid - y_max +
                                    height, x_min_valid - x_min:x_max_valid -
                                    x_max + width],
                   canvas[y_min_valid:y_max_valid + 1,
                          x_min_valid:x_max_valid + 1])

    def __call__(self, data_dict):
        w, h = get_image_size(data_dict['image'])

        if 'poly_meta' in data_dict.keys():
            ind = data_dict['poly_meta'].index('ignore_flag')
            ignore_flags = data_dict['poly_meta'].values[ind]
        else:
            ignore_flags = np.array([False] * len(data_dict['poly']))

        ignore_flags = self.find_invalid(data_dict['poly'], ignore_flags)

        data_dict['ocrdet_shrink_map'], ignore_flags = generate_kernel((w, h), data_dict['poly'], self.shrink_ratio, ignore_flags=ignore_flags)
        data_dict['ocrdet_shrink_mask'] = generate_effective_mask((w, h), data_dict['poly'], ignore_flags)

        data_dict['ocrdet_thr_map'], data_dict['ocrdet_thr_mask'] = self.generate_thr_map((h, w), data_dict['poly'], ignore_flags)

        data_dict['ocrdet_shrink_map'] = data_dict['ocrdet_shrink_map'][np.newaxis, ...]
        data_dict['ocrdet_thr_map'] = data_dict['ocrdet_thr_map'][np.newaxis, ...]

        if 'poly_meta' in data_dict.keys():
            data_dict['poly_meta'].values[ind] = ignore_flags

        return data_dict

    def __repr__(self):
        return 'DBEncode(shrink_ratio={}, thr_min={}, thr_max={}, min_short_size={})'.format(self.shrink_ratio, self.thr_min, self.thr_max, self.min_short_size)


@INTERNODE.register_module()
class DBMCEncode(DBEncode):
    def __init__(self, 
        num_classes=1,
        shrink_ratio=0.4,
        thr_min=0.3,
        thr_max=0.7,
        min_short_size=8,
        **kwargs):
        self.num_classes = num_classes
        self.shrink_ratio = shrink_ratio
        self.thr_min = thr_min
        self.thr_max = thr_max
        self.min_short_size = min_short_size

    def __call__(self, data_dict):
        assert data_dict['poly_meta'].have('class_id')

        labels = data_dict['poly_meta'].get('class_id')

        w, h = get_image_size(data_dict['image'])

        if 'poly_meta' in data_dict.keys():
            ind = data_dict['poly_meta'].index('ignore_flag')
            ignore_flags = data_dict['poly_meta'].values[ind]
        else:
            ignore_flags = np.array([False] * len(data_dict['poly']))

        ignore_flags = self.find_invalid(data_dict['poly'], ignore_flags)

        data_dict['ocrdet_shrink_map'] = []
        for i in range(self.num_classes):
            keep = np.nonzero(labels == i)[0].tolist()
            tmp_polys = [data_dict['poly'][k] for k in keep]
            tmp_flags = [ignore_flags[k] for k in keep]

            shrink_map, tmp_flags = generate_kernel((w, h), tmp_polys, self.shrink_ratio, ignore_flags=tmp_flags)
            data_dict['ocrdet_shrink_map'].append(shrink_map)

            for j, k in enumerate(keep):
                ignore_flags[k] = tmp_flags[j]
        data_dict['ocrdet_shrink_map'] = np.array(data_dict['ocrdet_shrink_map'])

        data_dict['ocrdet_shrink_mask'] = generate_effective_mask((w, h), data_dict['poly'], ignore_flags)

        data_dict['ocrdet_thr_map'], data_dict['ocrdet_thr_mask'] = self.generate_thr_map((h, w), data_dict['poly'], ignore_flags)
        data_dict['ocrdet_thr_map'] = data_dict['ocrdet_thr_map'][np.newaxis, ...]

        if 'poly_meta' in data_dict.keys():
            data_dict['poly_meta'].values[ind] = ignore_flags

        return data_dict

    def __repr__(self):
        return 'DBMCEncode(num_classes, shrink_ratio={}, thr_min={}, thr_max={}, min_short_size={})'.format(self.num_classes, self.shrink_ratio, self.thr_min, self.thr_max, self.min_short_size)
