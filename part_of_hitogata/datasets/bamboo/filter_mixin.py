from .builder import INTERNODE
from .builder import build_internode
from .base_internode import BaseInternode
from ..utils.common import clip_bbox, clip_poly, filter_bbox_by_length, filter_point, filter_poly, filter_list


class BaseFilterMixin(object):
    use_base_filter = True

    # def base_filter(self, data_dict):
    #     # print('fff', self.use_base_filter)
    #     if not self.use_base_filter:
    #         return data_dict

    #     if 'bbox' in data_dict.keys():
    #         keep = filter_bbox_by_length(data_dict['bbox'])
    #         data_dict['bbox'] = data_dict['bbox'][keep]

    #         if 'bbox_meta' in data_dict.keys():
    #             data_dict['bbox_meta']['keep'] = keep
    #             data_dict['bbox_meta'].filter(keep)

    #     if 'point' in data_dict.keys():
    #         keep_point, keep_instance = filter_point(data_dict['point'])
    #         data_dict['point'] = data_dict['point'][keep_instance]

    #         if 'point_meta' in data_dict.keys():
    #             data_dict['point_meta']['keep'] = keep_point
    #             data_dict['point_meta'].filter(keep_instance)

    #     if 'poly' in data_dict.keys():
    #         keep_poly = filter_poly(data_dict['poly'])
    #         data_dict['poly'] = filter_list(data_dict['poly'], keep_poly)

    #         if 'poly_meta' in data_dict.keys():
    #             data_dict['poly_meta']['keep'] = keep_poly
    #             data_dict['poly_meta'].filter(keep_poly)

    #     return data_dict

    def base_filter_bbox(self, data_dict):
        # print('fff', self.use_base_filter)
        if not self.use_base_filter:
            return data_dict

        target_tag = data_dict['intl_base_target_tag']

        keep = filter_bbox_by_length(data_dict[target_tag])
        data_dict[target_tag] = data_dict[target_tag][keep]

        if target_tag + '_meta' in data_dict.keys():
            data_dict[target_tag + '_meta']['keep'] = keep
            data_dict[target_tag + '_meta'].filter(keep)

        return data_dict

    def base_filter_point(self, data_dict):
        if not self.use_base_filter:
            return data_dict

        target_tag = data_dict['intl_base_target_tag']

        keep_point, keep_instance = filter_point(data_dict[target_tag])
        data_dict[target_tag] = data_dict[target_tag][keep_instance]

        if target_tag + '_meta' in data_dict.keys():
            data_dict[target_tag + '_meta']['keep'] = keep_point
            data_dict[target_tag + '_meta'].filter(keep_instance)

        return data_dict

    def base_filter_poly(self, data_dict):
        if not self.use_base_filter:
            return data_dict

        target_tag = data_dict['intl_base_target_tag']

        keep_poly = filter_poly(data_dict[target_tag])
        data_dict[target_tag] = filter_list(data_dict[target_tag], keep_poly)

        if target_tag + '_meta' in data_dict.keys():
            data_dict[target_tag + '_meta']['keep'] = keep_poly
            data_dict[target_tag + '_meta'].filter(keep_poly)

        return data_dict
