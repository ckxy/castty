import cv2
import math
import random
import numpy as np
from PIL import Image
from copy import deepcopy
from .base_internode import BaseInternode
from .builder import build_internode
from .builder import INTERNODE
from .bamboo import Bamboo
from ..utils.warp_tools import get_image_size, is_pil
from torchvision.transforms.functional import pad


__all__ = ['MixUp', 'CutMix', 'Mosaic']


@INTERNODE.register_module()
class MixUp(Bamboo):
    def __init__(self, internodes, **kwargs):
        assert len(internodes) > 0

        self.internodes = []
        for cfg in internodes:
            self.internodes.append(build_internode(cfg))

    @staticmethod
    def padding(image, right, bottom):
        if is_pil(image):
            return pad(image, (0, 0, right, bottom), 0, 'constant')
        else:
            # if image.ndim == 2:
            #     value = 0
            # else:
            #     value = tuple([0] * image.shape[2])
            return cv2.copyMakeBorder(image, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    def __call__(self, data_dict):
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

            w_a, h_a = get_image_size(a['image'])
            w_b, h_b = get_image_size(b['image'])
            max_w = max(w_a, w_b)
            max_h = max(h_a, h_b)

            a['image'] = self.padding(a['image'], max(max_w - w_a, 0), max(max_h - h_a, 0))
            b['image'] = self.padding(b['image'], max(max_w - w_b, 0), max(max_h - h_b, 0))

            a['ori_size'] = np.array([max_h, max_w]).astype(np.float32)

            lam = np.random.beta(1.5, 1.5)

            if is_pil(a['image']):
                a['image'] = Image.blend(a['image'], b['image'], 1 - lam)
            else:
                a['image'] = cv2.addWeighted(a['image'], lam, b['image'], 1 - lam, 0)

            if 'bbox' in k:
                # a['bbox'][:, 5] *= lam
                # b['bbox'][:, 5] *= 1 - lam
                a['bbox'] = np.concatenate([a['bbox'], b['bbox']])

                if 'bbox_meta' in k:
                    i = a['bbox_meta'].index('score')
                    a['bbox_meta'].values[i] *= lam
                    b['bbox_meta'].values[i] *= 1 - lam
                    a['bbox_meta'] += b['bbox_meta']

            if 'path' in k:
                a['path'] = '[mixup]({}, {}, {})'.format(a['path'], b['path'], lam)

            return a

    def reverse(self, **kwargs):
        return kwargs

    def rper(self):
        return type(self).__name__ + '(not available)'


class CutMix(Bamboo):
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


@INTERNODE.register_module()
class Mosaic(Bamboo):
    def __init__(self, internodes, **kwargs):
        assert len(internodes) > 0

        self.internodes = []
        for cfg in internodes:
            self.internodes.append(build_internode(cfg))

    def __call__(self, data_dict):
        res = dict()
        indices = [random.randint(0, data_dict['len_data_lines'] - 1) for _ in range(3)]
        # indices = [1, 2, 3]
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

        # 1 2
        # 3 4

        w1, h1 = get_image_size(d1['image'])
        w2, h2 = get_image_size(d2['image'])
        w3, h3 = get_image_size(d3['image'])
        w4, h4 = get_image_size(d4['image'])

        xc = max(w1, w3)
        yc = max(h1, h2)

        new_w = xc + max(w2, w4)
        new_h = yc + max(h3, h4)

        if is_pil(d1['image']):
            img4 = Image.new('RGB', (new_w, new_h), (0, 0, 0))

            img4.paste(d1['image'], (xc - w1, yc - h1))
            img4.paste(d2['image'], (xc, yc - h2))
            img4.paste(d3['image'], (xc - w3, yc))
            img4.paste(d4['image'], (xc, yc))
        else:
            img4 = np.zeros((new_h, new_w, 3)).astype(np.uint8)

            # print(img4[yc - h1:yc, xc - w1:xc].shape, d1['image'].shape)
            # print(img4[yc - h2:yc, xc:xc + w2].shape, d2['image'].shape)
            # print(img4[yc:yc + h3, xc - w3:xc].shape, d3['image'].shape)
            # print(img4[yc:yc + h4, xc:xc + w4].shape, d4['image'].shape)
            img4[yc - h1:yc, xc - w1:xc] = d1['image']
            img4[yc - h2:yc, xc:xc + w2] = d2['image']
            img4[yc:yc + h3, xc - w3:xc] = d3['image']
            img4[yc:yc + h4, xc:xc + w4] = d4['image']
            # exit()

        res['image'] = img4
        res['ori_size'] = np.array([new_h, new_w]).astype(np.float32)

        if 'bbox' in k:
            b1 = self.adjust_bbox(d1['bbox'], xc - w1, yc - h1)
            b2 = self.adjust_bbox(d2['bbox'], xc, yc - h2)
            b3 = self.adjust_bbox(d3['bbox'], xc - w3, yc)
            b4 = self.adjust_bbox(d4['bbox'], xc, yc)
            res['bbox'] = np.concatenate((b1, b2, b3, b4))

            if 'bbox_meta' in k:
                i = d1['bbox_meta'].index('score')
                d1['bbox_meta'] += d2['bbox_meta']
                d1['bbox_meta'] += d3['bbox_meta']
                d1['bbox_meta'] += d4['bbox_meta']
                res['bbox_meta'] = deepcopy(d1['bbox_meta'])

        if 'path' in k:
            res['path'] = '[mosaic]({}, {}, {}, {})'.format(d1['path'], d2['path'], d3['path'], d4['path'])

        return res

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

    def rper(self):
        return type(self).__name__ + '(not available)'

