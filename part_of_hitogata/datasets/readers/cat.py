from .reader import Reader
from .builder import READER, build_reader


__all__ = ['CatReader']


@READER.register_module()
class CatReader(Reader):
    def __init__(self, internodes, use_pil=True, **kwargs):
        assert len(internodes) > 1

        self.internodes = []
        for cfg in internodes:
            cfg['use_pil'] = use_pil
            self.internodes.append(build_reader(cfg))
            if len(self.internodes) > 1:
                assert self.internodes[-2].info['forcat'] == self.internodes[-1].info['forcat']

        self.groups = [0]
        for i in self.internodes:
            self.groups.append(len(i))

        for i in range(len(self.groups) - 1):
            self.groups[i + 1] += self.groups[i]

        self._info = dict()
        for i in self.internodes:
            self._info.update(i.info)

    def get_offset(self, index):
        for i in range(len(self.groups) - 1):
            if self.groups[i] <= index < self.groups[i + 1]:
                index -= self.groups[i]
                break
            elif index < self.groups[i]:
                raise ValueError
        return index, i

    def __call__(self, index):
        offset, gid = self.get_offset(index)
        res = self.internodes[gid](offset)
        return res

    def __len__(self):
        return self.groups[-1]

    def __repr__(self):
        res = 'CatReader(\n'
        for t in self.internodes:
            res += '  ' + t.__repr__() + '\n'
        res = res[:-1]
        res += '\n  )'
        return res
