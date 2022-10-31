from .builder import INTERNODE


__all__ = ['BaseInternode']


@INTERNODE.register_module()
class BaseInternode(object):
    def calc_intl_param_forward(self, data_dict):
        return data_dict

    def forward(self, data_dict):
        return data_dict

    def erase_intl_param_forward(self, data_dict):
        return data_dict

    def calc_intl_param_backward(self, data_dict):
        return data_dict

    def backward(self, data_dict):
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
