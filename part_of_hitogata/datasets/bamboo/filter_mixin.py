from ..utils.common import clip_bbox, clip_poly, filter_bbox_by_length, filter_point, filter_poly, filter_list


class BaseFilterMixin(object):
    use_base_filter = True

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
