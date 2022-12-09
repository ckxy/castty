from .builder import INTERNODE
from .builder import build_internode
from .base_internode import BaseInternode


__all__ = ['InternodeWithFilter']


@INTERNODE.register_module()
class InternodeWithFilter(BaseInternode):
    def __init__(self, filter_cfg, **kwargs):
        self.filter_internodes = []
        for cfg in filter_cfg:
            self.filter_internodes.append(build_internode(cfg))

    def clip_and_filter(self, data_dict):
        for t in self.filter_internodes:
            data_dict = t(data_dict)
        return data_dict

    def __call__(self, data_dict):
        data_dict = self.calc_intl_param_forward(data_dict)
        data_dict = self.forward(data_dict)
        data_dict = self.clip_and_filter(data_dict)
        data_dict = self.erase_intl_param_forward(data_dict)
        return data_dict

    def reverse(self, **kwargs):
        kwargs = self.calc_intl_param_backward(kwargs)
        kwargs = self.backward(kwargs)
        kwargs = self.clip_and_filter(kwargs)
        kwargs = self.erase_intl_param_backward(kwargs)
        return kwargs
