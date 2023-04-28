from .reader import Reader
from .builder import READER, build_reader


__all__ = ['CatReader']


@READER.register_module()
class CatReader(Reader):
    def __init__(self, readers, use_pil=True, output_gid=False, **kwargs):
        assert len(readers) > 1

        self.output_gid = output_gid

        self.readers = []
        for cfg in readers:
            assert cfg['type'] != 'CatReader'
            cfg['use_pil'] = use_pil
            self.readers.append(build_reader(cfg))
            if len(self.readers) > 1:
                assert self.readers[-2].info['forcat'] == self.readers[-1].info['forcat']

        self.groups = [0]
        for i in self.readers:
            self.groups.append(len(i))

        for i in range(len(self.groups) - 1):
            self.groups[i + 1] += self.groups[i]

        self._info = dict()
        for i in self.readers:
            self._info.update(i.info)

    def get_offset(self, index):
        for i in range(len(self.groups) - 1):
            if self.groups[i] <= index < self.groups[i + 1]:
                index -= self.groups[i]
                break
            elif index < self.groups[i]:
                raise ValueError
        return index, i

    def __getitem__(self, index):
        offset, gid = self.get_offset(index)
        res = self.readers[gid][offset]
        if self.output_gid:
            res['intl_group_id'] = gid
        return res

    def __len__(self):
        return self.groups[-1]

    def __repr__(self):
        res = 'CatReader(\n'
        for t in self.readers:
            res += '  ' + t.__repr__() + '\n'
        res += '  output_gid={}'.format(self.output_gid)
        res += '\n  )'
        return res
