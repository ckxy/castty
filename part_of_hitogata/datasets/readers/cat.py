from .reader import Reader
from addict import Dict
from .builder import READER, build_reader


__all__ = ['CatReader']


@READER.register_module()
class CatReader(Reader):
    def __init__(self, internodes):
        assert len(internodes) > 0

        self.internodes = []
        for cfg in internodes:
            self.internodes.append(build_reader(cfg))
            if len(self.internodes) > 1:
                assert self.internodes[-2].get_dataset_info()[1] == self.internodes[-1].get_dataset_info()[1]

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
        offset, gid = self.get_offset(index)
        res = self.internodes[gid](offset)
        return res

    def __repr__(self):
        res = 'CatReader(\n'
        for t in self.internodes:
            res += '  ' + t.__repr__() + '\n'
        res = res[:-1]
        res += '\n  )'
        return res
