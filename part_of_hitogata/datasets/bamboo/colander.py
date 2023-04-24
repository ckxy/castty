import numpy as np
from .builder import INTERNODE
from ...utils.bbox_tools import xyxy2xywh
from .base_internode import BaseInternode
from ..utils.common import get_image_size, clip_bbox, clip_poly
try:
    from shapely.geometry import Polygon
except ImportError:
    pass


__all__ = ['FilterBboxByLength', 'FilterBboxByArea', 'FilterBboxByLengthRatio', 'FilterBboxByAreaRatio', 'FilterBboxByAspectRatio', 'FilterSelfOverlapping']


class Colander(BaseInternode):
    def calc_keep(self, data_dict):
        raise NotImplementedError


class BBoxColander(Colander):
    def forward(self, data_dict):
        if 'bbox' in data_dict.keys():
            keep = self.calc_keep(data_dict)
            data_dict['bbox'] = data_dict['bbox'][keep]

            if 'bbox_meta' in data_dict.keys():
                data_dict['bbox_meta'].filter(keep)
        return data_dict

    def backward(self, data_dict):
        return self.forward(data_dict)


@INTERNODE.register_module()
class FilterBboxByLength(BBoxColander):
    def __init__(self, min_w=1, min_h=1):
        assert min_w >= 1 and min_h >= 1

        self.min_w = min_w
        self.min_h = min_h

    def calc_keep(self, data_dict):
        xywh = xyxy2xywh(data_dict['bbox'].copy())
        keep = (xywh[..., 2] >= self.min_w) * (xywh[..., 3] >= self.min_h)
        keep = np.nonzero(keep)[0].tolist()
        return keep

    def __repr__(self):
        return 'FilterBboxByLength(min_w={}, min_h={})'.format(self.min_w, self.min_h)


@INTERNODE.register_module()
class FilterBboxByArea(BBoxColander):
    def __init__(self, min_a=0):
        assert min_a > 0
        self.min_a = min_a

    def calc_keep(self, data_dict):
        xywh = xyxy2xywh(data_dict['bbox'].copy())
        area = xywh[..., 2] * xywh[..., 3]
        keep = area >= self.min_a
        keep = np.nonzero(keep)[0].tolist()
        return keep

    def __repr__(self):
        return 'FilterBboxByArea(min_a={})'.format(self.min_a)


@INTERNODE.register_module()
class FilterBboxByLengthRatio(BBoxColander):
    def __init__(self, min_w=None, min_h=None):
        assert 0 <= min_w < 1 and 0 <= min_h < 1
        assert min_w > 0 or min_h > 0

        self.min_w = min_w
        self.min_h = min_h

    def calc_keep(self, data_dict):
        xywh = xyxy2xywh(data_dict['bbox'].copy())
        w, h = get_image_size(data_dict['image'])
        keep = (xywh[..., 2] >= (self.min_w * w)) * (xywh[..., 3] >= (self.min_h * h))
        keep = np.nonzero(keep)[0].tolist()
        return keep

    def __repr__(self):
        return 'FilterBboxByLengthRatio(min_w={}, min_h={})'.format(self.min_w, self.min_h)


@INTERNODE.register_module()
class FilterBboxByAreaRatio(BBoxColander):
    def __init__(self, min_a=0):
        assert min_a > 0
        self.min_a = min_a

    def calc_keep(self, data_dict):
        xywh = xyxy2xywh(data_dict['bbox'].copy())
        area = xywh[..., 2] * xywh[..., 3]
        w, h = get_image_size(data_dict['image'])
        keep = area >= (self.min_a * w * h)
        keep = np.nonzero(keep)[0].tolist()
        return keep

    def __repr__(self):
        return 'FilterBboxByAreaRatio(min_a={})'.format(self.min_a)


@INTERNODE.register_module()
class FilterBboxByAspectRatio(BBoxColander):
    def __init__(self, aspect_ratio):
        assert isinstance(aspect_ratio, tuple) and len(aspect_ratio) == 2
        assert aspect_ratio[0] > 0 and aspect_ratio[1] > 0 and aspect_ratio[0] <= aspect_ratio[1]
        self.aspect_ratio = aspect_ratio

    def calc_keep(self, data_dict):
        xywh = xyxy2xywh(data_dict['bbox'].copy())
        keep = (xywh[..., 2] <= (xywh[..., 3] * self.aspect_ratio[1])) * (xywh[..., 2] >= (xywh[..., 3] * self.aspect_ratio[0]))
        keep = np.nonzero(keep)[0].tolist()
        return keep

    def __repr__(self):
        return 'FilterBboxByAspectRatio(aspect_ratio={})'.format(self.aspect_ratio)


@INTERNODE.register_module()
class FilterSelfOverlapping(BaseInternode):
    def __init__(self, iou=None):
        self.iou = iou

    def forward(self, data_dict):
        if 'poly' in data_dict.keys() and len(data_dict['poly']) > 0:
            polygon_shapes = [Polygon(p) for p in data_dict['poly']]

            reject = set()
            for i in range(len(data_dict['poly']) - 1):
                for j in range(i + 1, len(data_dict['poly'])):
                    if self.iou is None:
                        if polygon_shapes[i].intersects(polygon_shapes[j]):
                            reject.add(i)
                            reject.add(j)
                    else:
                        intersect = polygon1.intersection(polygon2).area
                        union = polygon1.union(polygon2).area + 1e-6
                        iou = intersect / union

                        if iou >= self.iou:
                            reject.add(i)
                            reject.add(j)
            
            if 'poly_meta' in data_dict.keys():
                ignore_flags = data_dict['poly_meta']['ignore_flag']
                for r in reject:
                    ignore_flags[r] = True
                data_dict['poly_meta']['ignore_flag'] = ignore_flags
            else:
                keep = set(range(len(data_dict['poly']))) - reject
                keep = sorted(list(keep))
                data_dict['poly'] = [data_dict['poly'][k] for k in keep]
        return data_dict

    def __repr__(self):
        return 'FilterSelfOverlapping(iou={})'.format(self.iou)

