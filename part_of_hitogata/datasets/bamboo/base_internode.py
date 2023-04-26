from .builder import INTERNODE


__all__ = ['BaseInternode']


TAG_MAPPING = dict(
    image=['image'],
    label=['label'],
    bbox=['bbox'],
    mask=['mask'],
    point=['point'],
    poly=['poly'],
    seq=['seq']
)


@INTERNODE.register_module()
class BaseInternode(object):
    def __init__(self, tag_mapping=TAG_MAPPING, **kwargs):
        self.tag_mapping = tag_mapping

    def calc_intl_param_forward(self, data_dict):
        return data_dict

    def forward_image(self, data_dict):
        return data_dict

    def forward_label(self, data_dict):
        return data_dict

    def forward_bbox(self, data_dict):
        return data_dict

    def forward_mask(self, data_dict):
        return data_dict

    def forward_point(self, data_dict):
        return data_dict

    def forward_poly(self, data_dict):
        return data_dict

    def forward_seq(self, data_dict):
        return data_dict

    def forward(self, data_dict):
        for map2func, tags in self.tag_mapping.items():
            for tag in tags:
                if tag in data_dict.keys():
                    data_dict['intl_base_target_tag'] = tag
                    data_dict = getattr(self, 'forward_{}'.format(tag))(data_dict)
                    data_dict.pop('intl_base_target_tag')
        return data_dict

    def erase_intl_param_forward(self, data_dict):
        return data_dict

    def calc_intl_param_backward(self, data_dict):
        return data_dict

    def backward_image(self, data_dict):
        return data_dict

    def backward_label(self, data_dict):
        return data_dict

    def backward_bbox(self, data_dict):
        return data_dict

    def backward_mask(self, data_dict):
        return data_dict

    def backward_point(self, data_dict):
        return data_dict

    def backward_poly(self, data_dict):
        return data_dict

    def backward_seq(self, data_dict):
        return data_dict

    def backward(self, data_dict):
        for map2func, tags in self.tag_mapping.items():
            for tag in tags:
                if tag in data_dict.keys():
                    data_dict['intl_base_target_tag'] = tag
                    data_dict = getattr(self, 'backward_{}'.format(tag))(data_dict)
                    data_dict.pop('intl_base_target_tag')
        return data_dict

    def erase_intl_param_backward(self, data_dict):
        return data_dict

    def __call__(self, data_dict):
        data_dict = self.calc_intl_param_forward(data_dict)
        data_dict = self.forward(data_dict)
        data_dict = self.erase_intl_param_forward(data_dict)
        return data_dict

    def reverse(self, **kwargs):
        kwargs = self.calc_intl_param_backward(kwargs)
        kwargs = self.backward(kwargs)
        kwargs = self.erase_intl_param_backward(kwargs)
        return kwargs

    def __repr__(self):
        return type(self).__name__ + '()'

    def rper(self):
        return type(self).__name__ + '(not available)'
