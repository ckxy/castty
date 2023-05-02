import torch
import random
import numpy as np
from copy import deepcopy
from ..utils.registry import Registry, build_from_cfg
from torch.utils.data._utils.collate import default_collate


COLLATEFN = Registry('collatefn')


def build_collatefn(cfg, **default_args):
    return build_from_cfg(cfg, COLLATEFN, default_args)


class Collator(object):
    def __init__(self, fn_list):
        self.fn_list = []
        for cfg in fn_list:
            self.fn_list.append(build_collatefn(cfg))

    def collate_fn(self, batch):
        # print(sorted(batch[0].keys()), len(batch[0].keys()))
        # exit()

        for i in range(len(batch)):
            for fn in self.fn_list:
                batch[i] = fn(batch[i])

        # print(sorted(batch[0].keys()), len(batch[0].keys()))
        # exit()

        res = default_collate(batch)

        for fn in self.fn_list:
            tmp = fn.collate()
            res.update(tmp)

        # print(sorted(res.keys()), len(res.keys()))
        # exit()
        return res

    def __repr__(self):
        if len(self.fn_list) == 0:
            return 'Collator(None)'
        else:
            res = 'Collator(\n'
            for t in self.fn_list:
                res += '  ' + t.__repr__() + '\n'
            res = res[:-1]
            res += '\n)'
            return res


class CollateFN(object):
    def __init__(self, **kwargs):
        assert isinstance(kwargs['names'], tuple) or isinstance(kwargs['names'], list)
        assert len(kwargs['names']) > 0

        self.names = kwargs['names']
        self.buffer = dict()
        for name in self.names:
            self.buffer[name] = []

    def __call__(self, data_dict):
        for name in self.names:
            if name in data_dict.keys():
                tmp = data_dict.pop(name)
                self.buffer[name].append(tmp)
            else:
                raise KeyError
        return data_dict

    def collate(self):
        raise NotImplementedError

    def __repr__(self):
        return 'CollateFN(names={})'.format(self.names)


@COLLATEFN.register_module()
class ListCollateFN(CollateFN):
    def collate(self):
        res = dict()
        for k in self.buffer.keys():
            res[k] = deepcopy(self.buffer[k])
            self.buffer[k].clear()
        return res

    def __repr__(self):
        return 'ListCollateFN(names={})'.format(self.names)


@COLLATEFN.register_module()
class BboxCollateFN(CollateFN):
    def collate(self):
        res = dict()
        for k in self.buffer.keys():
            res[k] = [torch.from_numpy(b).type(torch.float32) for b in self.buffer[k]]
            self.buffer[k].clear()
        return res

    def __repr__(self):
        return 'BboxCollateFN(names={})'.format(self.names)


@COLLATEFN.register_module()
class EnSeqCollateFN(CollateFN):
    def collate(self):
        res = dict()
        for k in self.buffer.keys():
            res[k] = torch.cat(self.buffer[k])
            self.buffer[k].clear()
        return res

    def __repr__(self):
        return 'EnSeqCollateFN(names={})'.format(self.names)


@COLLATEFN.register_module()
class LabelCollateFN(CollateFN):
    def collate(self):
        res = dict()
        for k in self.buffer.keys():
            res[k] = []
            for i in range(len(self.buffer[k])):
                if len(self.buffer[k][0]) > 1:
                    t = []
                    for j in range(len(self.buffer[k][0])):
                        t.append(torch.from_numpy(self.buffer[k][i][j]).type(torch.float32))
                    res[k].append(t)
                else:
                    res[k].append(torch.from_numpy(self.buffer[k][i][0]).type(torch.float32))
            self.buffer[k].clear()
        return res

    def __repr__(self):
        return 'LabelCollateFN(names={})'.format(self.names)


@COLLATEFN.register_module()
class NanoCollateFN(CollateFN):
    def __init__(self, **kwargs):
        super(NanoCollateFN, self).__init__(names=('nano_fs', 'nano_grid', 'nano_pnc', 'nano_target', 'num_neg', 'num_pos'))

    def collate(self):
        targets = [t.unsqueeze(0) for t in self.buffer['nano_target']]
        grids = [t.unsqueeze(0) for t in self.buffer['nano_grid']]
        targets = torch.cat(targets)
        grids = torch.cat(grids)

        featmap_sizes = self.buffer['nano_fs'][0]
        nums = [i[0] * i[1] for i in featmap_sizes]

        targets = torch.split(targets, nums, dim=1)
        grids = torch.split(grids, nums, dim=1)

        num_pos = sum(self.buffer['num_pos'])
        num_neg = sum(self.buffer['num_neg'])

        pnc = np.array(self.buffer['nano_pnc'])
        pnc = np.sum(pnc, axis=0)

        for k in self.buffer.keys():
            self.buffer[k].clear()

        return dict(
            nano_grid=grids,
            nano_pnc=pnc,
            nano_target=targets,
            num_neg=num_neg,
            num_pos=num_pos
        )

    def __repr__(self):
        return 'NanoCollateFN(names={})'.format(self.names)

if __name__ == '__main__':
    pass
