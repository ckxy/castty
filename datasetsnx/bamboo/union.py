import math
import random
import numpy as np
from PIL import Image
from copy import deepcopy
from .base_internode import BaseInternode
from .bambusa import Bambusa

from .misc import *

from .color import *
from .convert import *
from .crop import *
from .erase import *
from .filp import *
from .image import *
from .label import *
from .pad import *
from .register import *
from .resize import *
from .tag import *
from .warp import *


__all__ = ['ChooseOne', 'MixUp', 'CutMix', 'Mosaic']


class ChooseOne(BaseInternode):
    def __init__(self, internodes):
        self.internodes = []
        for k, v in internodes:
            self.internodes.append(eval(k)(**v))

    def __call__(self, data_dict):
        i = random.choice(self.internodes)
        # i = self.internodes[0]
        # print(i)
        return i(data_dict)

    def __repr__(self):
        split_str = [i.__repr__() for i in self.internodes]
        bamboo_str = ''
        for i in range(len(split_str)):
            bamboo_str += '\n  ' + split_str[i].replace('\n', '\n  ')
        bamboo_str = '(\n{}\n  )'.format(bamboo_str[1:])

        return 'ChooseOne(\n  internodes: {} \n)'.format(bamboo_str)

    def rper(self):
        return 'ChooseOne(not available)'


class MixUp(Bambusa):
    def __init__(self, internodes, b=1.5, p=1):
        assert len(internodes) > 0
        assert 0 < p <= 1
        assert b > 0

        self.b = b
        self.p = p

        self.internodes = []
        for k, v in internodes:
            self.internodes.append(eval(k)(**v))

    def __call__(self, data_dict):
        if random.random() < self.p:
            index_mix = random.randint(0, data_dict['len_data_lines'] - 1)

            if index_mix == data_dict['index']:
                return super(MixUp, self).__call__(data_dict)
            else:
                data_dict_b = deepcopy(data_dict)
                data_dict_b['index'] = index_mix
                a = super(MixUp, self).__call__(data_dict)

                k = a.keys()
                assert 'mask' not in k and 'point' not in  k

                b = super(MixUp, self).__call__(data_dict_b)

                assert a['image'].size == b['image'].size

                lam = np.random.beta(self.b, self.b)

                a['image'] = Image.blend(a['image'], b['image'], 1 - lam)

                if 'bbox' in k:
                    a['bbox'][:, 5] *= lam
                    b['bbox'][:, 5] *= 1 - lam
                    a['bbox'] = np.concatenate([a['bbox'], b['bbox']])

                if 'path' in k:
                    a['path'] = a['path'] + '-%%-' + str(lam) + '-%%-' + b['path']

                if 'difficult' in k:
                    a['difficult'] = np.concatenate([a['difficult'], b['difficult']])

                if 'label' in k:
                    a['label_add'] = b['label']
                    a['lam'] = lam

                return a
        else:
            return super(MixUp, self).__call__(data_dict)

    def __repr__(self):
        split_str = [i.__repr__() for i in self.internodes]
        bamboo_str = ''
        for i in range(len(split_str)):
            bamboo_str += '\n  ' + split_str[i].replace('\n', '\n  ')
        bamboo_str = '(\n{}\n  )'.format(bamboo_str[1:])

        return 'MixUp(\n  p: {}\n  b: {}\n  internodes: {} \n)'.format(self.p, self.b, bamboo_str)

    def rper(self):
        return 'MixUp(not available)'


class CutMix(Bambusa):
    def __init__(self, internodes, b=1.5, p=1):
        assert len(internodes) > 0
        assert 0 < p <= 1
        assert b > 0

        self.b = b
        self.p = p

        self.internodes = []
        for k, v in internodes:
            self.internodes.append(eval(k)(**v))

    def __call__(self, data_dict):
        if random.random() < self.p:
            index_mix = random.randint(0, data_dict['len_data_lines'] - 1)

            if index_mix == data_dict['index']:
                return super(CutMix, self).__call__(data_dict)
            else:
                data_dict_b = deepcopy(data_dict)
                data_dict_b['index'] = index_mix
                a = super(CutMix, self).__call__(data_dict)

                k = a.keys()
                assert 'bbox' not in k and 'point' not in  k

                b = super(CutMix, self).__call__(data_dict_b)

                assert a['image'].size == b['image'].size

                lam = np.random.beta(self.b, self.b)

                bbx1, bby1, bbx2, bby2 = self.rand_bbox(a['image'].size, lam)
                a['image'].paste(b['image'].crop((bbx1, bby1, bbx2, bby2)), (bbx1, bby1, bbx2, bby2))
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (a['image'].size[0] * a['image'].size[1]))

                if 'path' in k:
                    a['path'] = a['path'] + '-**-' + str(lam) + '-**-' + b['path']

                if 'label' in k:
                    a['label_add'] = b['label']
                    a['lam'] = lam

                if 'mask' in k:
                    a['mask'].paste(b['mask'].crop((bbx1, bby1, bbx2, bby2)), (bbx1, bby1, bbx2, bby2))

                return a
        else:
            return super(CutMix, self).__call__(data_dict)

    @staticmethod
    def rand_bbox(size, lam):
        W, H = size
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __repr__(self):
        split_str = [i.__repr__() for i in self.internodes]
        bamboo_str = ''
        for i in range(len(split_str)):
            bamboo_str += '\n  ' + split_str[i].replace('\n', '\n  ')
        bamboo_str = '(\n{}\n  )'.format(bamboo_str[1:])

        return 'CutMix(\n  p: {}\n  b: {}\n  internodes: {} \n)'.format(self.p, self.b, bamboo_str)

    def rper(self):
        return 'CutMix(not available)'


class Mosaic(Bambusa):
    def __init__(self, internodes, p=1):
        assert len(internodes) > 0
        assert 0 < p <= 1

        self.p = p

        self.internodes = []
        for k, v in internodes:
            self.internodes.append(eval(k)(**v))

    def __call__(self, data_dict):
        if random.random() < self.p:
            res = dict()
            indices = [random.randint(0, data_dict['len_data_lines'] - 1) for _ in range(3)]
            tmp = deepcopy(data_dict)

            # print(data_dict['index'])
            d1 = super(Mosaic, self).__call__(data_dict)

            k = d1.keys()
            assert 'label' not in k

            t = deepcopy(tmp)
            t['index'] = indices[0]
            d2 = super(Mosaic, self).__call__(t)
            t = deepcopy(tmp)
            t['index'] = indices[1]
            d3 = super(Mosaic, self).__call__(t)
            t = deepcopy(tmp)
            t['index'] = indices[2]
            d4 = super(Mosaic, self).__call__(t)

            del tmp

            xc = max(d1['image'].size[0], d3['image'].size[0])
            yc = max(d1['image'].size[1], d2['image'].size[1])

            new_w = xc + max(d2['image'].size[0], d4['image'].size[0])
            new_h = yc + max(d3['image'].size[1], d4['image'].size[1])

            img4 = Image.new('RGB', (new_w, new_h), (0, 0, 0))

            img4.paste(d1['image'], (xc - d1['image'].size[0], yc - d1['image'].size[1]))
            img4.paste(d2['image'], (xc, yc - d2['image'].size[1]))
            img4.paste(d3['image'], (xc - d3['image'].size[0], yc))
            img4.paste(d4['image'], (xc, yc))

            res['image'] = img4
            res['ori_size'] = np.array([new_h, new_w]).astype(np.float32)

            if 'mask' in k:
                mask4 = Image.new('P', (new_w, new_h), 0)

                mask4.paste(d1['mask'], (xc - d1['mask'].size[0], yc - d1['mask'].size[1]))
                mask4.paste(d2['mask'], (xc, yc - d2['mask'].size[1]))
                mask4.paste(d3['mask'], (xc - d3['mask'].size[0], yc))
                mask4.paste(d4['mask'], (xc, yc))

                res['mask'] = mask4

            if 'bbox' in k:
                b1 = self.adjust_bbox(d1['bbox'], xc - d1['image'].size[0], yc - d1['image'].size[1])
                b2 = self.adjust_bbox(d2['bbox'], xc, yc - d2['image'].size[1])
                b3 = self.adjust_bbox(d3['bbox'], xc - d3['image'].size[0], yc)
                b4 = self.adjust_bbox(d4['bbox'], xc, yc)
                res['bbox'] = np.concatenate((b1, b2, b3, b4))

            if 'difficult' in k:
                res['difficult'] = np.concatenate((d1['difficult'], d2['difficult'], d3['difficult'], d4['difficult']))

            if 'point' in k:
                p1 = self.adjust_point(d1['point'], xc - d1['image'].size[0], yc - d1['image'].size[1])
                p2 = self.adjust_point(d2['point'], xc, yc - d2['image'].size[1])
                p3 = self.adjust_point(d3['point'], xc - d3['image'].size[0], yc)
                p4 = self.adjust_point(d4['point'], xc, yc)
                res['point'] = np.concatenate((p1, p2, p3, p4))

            if 'path' in k:
                res['path'] = d1['path'] + '-<>-' + d2['path'] + '-<>-' + d3['path'] + '-<>-' + d4['path']

            return res
        else:
            return super(Mosaic, self).__call__(data_dict)

    @staticmethod
    def adjust_bbox(bbox, ox, oy):
        bbox[:, 0] += ox
        bbox[:, 1] += oy
        bbox[:, 2] += ox
        bbox[:, 3] += oy
        return bbox

    @staticmethod
    def adjust_point(point, ox, oy):
        point[:, 0] += ox
        point[:, 1] += oy
        return point

    def __repr__(self):
        split_str = [i.__repr__() for i in self.internodes]
        bamboo_str = ''
        for i in range(len(split_str)):
            bamboo_str += '\n  ' + split_str[i].replace('\n', '\n  ')
        bamboo_str = '(\n{}\n  )'.format(bamboo_str[1:])

        return 'Mosaic(\n  p: {}\n  internodes: {} \n)'.format(self.p, bamboo_str)

    def rper(self):
        return 'Mosaic(not available)'
