from .builder import INTERNODE


__all__ = ['BaseInternode']


@INTERNODE.register_module()
class BaseInternode(object):
    def __call__(self, data_dict):
        return data_dict

    def reverse(self, **kwargs):
        return kwargs

    def __repr__(self):
        return type(self).__name__ + '()'

    def rper(self):
        return type(self).__name__ + '(not available)'
