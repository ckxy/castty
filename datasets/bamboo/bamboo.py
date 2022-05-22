from .base_internode import BaseInternode
from .builder import build_internode
from .builder import INTERNODE


@INTERNODE.register_module()
class Bamboo(BaseInternode):
    def __init__(self, internodes, **kwargs):
        assert len(internodes) > 0

        self.internodes = []
        for cfg in internodes:
            self.internodes.append(build_internode(cfg))

    def __call__(self, data_dict):
        for t in self.internodes:
            data_dict = t(data_dict)
        return data_dict

    def reverse(self, **kwargs):
        for t in self.internodes[::-1]:
            kwargs = t.reverse(**kwargs)
        return kwargs

    def __repr__(self):
        split_str = [i.__repr__() for i in self.internodes]
        bamboo_str = type(self).__name__ + '('
        for i in range(len(split_str)):
            bamboo_str += '\n  ' + split_str[i].replace('\n', '\n  ')
        bamboo_str = '{}\n)'.format(bamboo_str)

        return bamboo_str

    def rper(self):
        if len(self.internodes) == 0:
            return '(None)'
        else:
            res = 'Oobmab(\n'
            for t in self.internodes[::-1]:
                res += '  ' + t.rper() + '\n'
            res = res[:-1]
            res += '\n)'
            return res

    def have_internode(self, name):
        names = [type(i).__name__ for i in self.internodes]
        return name in names
