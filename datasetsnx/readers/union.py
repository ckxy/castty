from .reader import Reader
from . import *
from addict import Dict


__all__ = ['UniReader']


class UniReader(Reader):
    def __init__(self, internodes):
        self.internodes = []
        for k, v in internodes:
            if 'Reader' not in k:
                raise ValueError
            self.internodes.append(eval(k)(**v))

        self.groups = [0]
        for i in self.internodes:
            line, _ = i.get_dataset_info()
            self.groups.append(len(line))

        for i in range(len(self.groups) - 1):
            self.groups[i + 1] += self.groups[i]

    def get_offset(self, index):
        for i in range(len(self.groups) - 1):
            if self.groups[i] <= index < self.groups[i + 1]:
                index -= self.groups[i]
                break
            elif index < self.groups[i]:
                raise ValueError
        return index, i

    def get_dataset_info(self):
        lines = 0
        infos = dict()
        for i in self.internodes:
            line, info = i.get_dataset_info()
            infos.update(info)
            lines += len(line)
        return range(lines), Dict(infos)

    def get_data_info(self, index):
        offset, gid = self.get_offset(index)
        return self.internodes[gid].get_data_info(offset)

    def __call__(self, index):
        # index = data_dict
        offset, gid = self.get_offset(index)
        res = self.internodes[gid](offset)
        # print(index, gid, offset, res['path'])
        return res

    def __repr__(self):
        res = 'UniReader(\n'
        for t in self.internodes:
            res += '  ' + t.__repr__() + '\n'
        res = res[:-1]
        res += '\n  )'
        return res
