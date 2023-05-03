from ..utils.common import TAG_MAPPING, clip_bbox, clip_poly, filter_bbox_by_length, filter_point, filter_poly, filter_list


def identity(item, meta=None, **param):
    return item, meta


class DataAugMixin(object):
    def __init__(self, tag_mapping=None, forward_mapping=None, backward_mapping=None, **kwargs):
        if tag_mapping is None:
            self.tag_mapping = TAG_MAPPING
        else:
            self.tag_mapping = tag_mapping

        if forward_mapping is None:
            self.forward_mapping = dict()
            for tag in self.tag_mapping.keys():
                self.forward_mapping[tag] = identity
        else:
            self.forward_mapping = forward_mapping

        if backward_mapping is None:
            self.backward_mapping = dict()
            for tag in self.tag_mapping.keys():
                self.backward_mapping[tag] = identity
        else:
            self.backward_mapping = backward_mapping

    def forward(self, data_dict, **param):
        for map2func, tags in self.tag_mapping.items():
            if map2func not in self.forward_mapping.keys():
                continue
            for tag in tags:
                if tag in data_dict.keys():
                    item, meta = self.forward_mapping[tag](data_dict[tag], data_dict.get(tag + '_meta', None), **param)
                    data_dict[tag] = item
                    if meta is not None:
                        data_dict[tag + '_meta'] = meta
        return data_dict

    def backward(self, data_dict, **param):
        for map2func, tags in self.tag_mapping.items():
            if map2func not in self.backward_mapping.keys():
                continue
            for tag in tags:
                if tag in data_dict.keys():
                    item, meta = self.backward_mapping[tag](data_dict[tag], data_dict.get(tag + '_meta', None), **param)
                    data_dict[tag] = item
                    if meta is not None:
                        data_dict[tag + '_meta'] = meta
        return data_dict


class BaseFilterMixin(object):
    use_base_filter = True

    def base_filter_bbox(self, bbox, meta=None):
        # print(self.use_base_filter, 'fff')
        if not self.use_base_filter:
            return bbox, meta

        keep = filter_bbox_by_length(bbox)
        bbox = bbox[keep]

        if meta is not None:
            meta['keep'] = keep
            meta.filter(keep)

        return bbox, meta

    def base_filter_point(self, point, meta=None):
        if not self.use_base_filter:
            return point, meta

        keep_point, keep_instance = filter_point(point)
        point = point[keep_instance]

        if meta is not None:
            meta['keep'] = keep_point
            meta.filter(keep_instance)

        return point, meta

    def base_filter_poly(self, poly, meta=None):
        if not self.use_base_filter:
            return poly, meta

        keep_poly = filter_poly(poly)
        poly = filter_list(poly, keep_poly)

        if meta is not None:
            # print(meta['keep'], keep_poly)
            meta['keep'] = keep_poly
            meta.filter(keep_poly)

        return poly, meta
