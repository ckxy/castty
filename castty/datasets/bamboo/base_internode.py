from .builder import INTERNODE


__all__ = ['BaseInternode']


@INTERNODE.register_module()
class BaseInternode(object):
    def __init__(self, **kwargs):
        pass

    def calc_intl_param_forward(self, data_dict):
        return dict()

    def forward(self, data_dict, **kwargs):
        return data_dict

    def forward_rest(self, data_dict, **kwargs):
        return data_dict

    def calc_intl_param_backward(self, data_dict):
        return dict()

    def backward(self, data_dict, **kwargs):
        return data_dict

    def backward_rest(self, data_dict, **kwargs):
        return data_dict

    def __call__(self, data_dict):
        param = self.calc_intl_param_forward(data_dict)
        data_dict = self.forward(data_dict, **param)
        data_dict = self.forward_rest(data_dict, **param)
        return data_dict

    def reverse(self, **kwargs):
        param = self.calc_intl_param_backward(kwargs)
        kwargs = self.backward(kwargs, **param)
        kwargs = self.backward_rest(kwargs, **param)
        return kwargs

    def __repr__(self):
        return type(self).__name__ + '()'

    def rper(self):
        return type(self).__name__ + '(not available)'
