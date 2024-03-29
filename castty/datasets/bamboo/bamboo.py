from .builder import INTERNODE
from .builder import build_internode
from .base_internode import BaseInternode


__all__ = ['Bamboo']


@INTERNODE.register_module()
class Bamboo(BaseInternode):
    def __init__(self, internodes, **kwargs):
        assert len(internodes) > 0

        self.internodes = []
        for cfg in internodes:
            self.internodes.append(build_internode(cfg, **kwargs))

        BaseInternode.__init__(self, **kwargs)

    def forward(self, data_dict):
        for t in self.internodes:
            data_dict = t(data_dict)
        return data_dict

    def backward(self, data_dict):
        for t in self.internodes[::-1]:
            data_dict = t.reverse(**data_dict)
        return data_dict

    def __repr__(self):
        split_str = [i.__repr__() for i in self.internodes]
        bamboo_str = type(self).__name__ + '('
        for i in range(len(split_str)):
            bamboo_str += '\n  ' + split_str[i].replace('\n', '\n  ')
        bamboo_str += '\n)'

        return bamboo_str

    def rper(self):
        res = type(self).__name__
        res = res[:1].lower() + res[1:]
        res = res[::-1]
        res = res[:1].upper() + res[1:] + '('

        split_str = [i.rper() for i in self.internodes[::-1]]

        for i in range(len(split_str)):
            res += '\n  ' + split_str[i].replace('\n', '\n  ')
        res += '\n)'
        return res

    def have_internode(self, name):
        names = [type(i).__name__ for i in self.internodes]
        return name in names
