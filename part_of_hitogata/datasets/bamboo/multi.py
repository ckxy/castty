import cv2
import math
import random
import numpy as np
from PIL import Image
from copy import deepcopy
from .bamboo import Bamboo
from .builder import INTERNODE
from .builder import build_internode
from .base_internode import BaseInternode
from ..utils.common import get_image_size, is_pil
from torchvision.transforms.functional import pad

from .pad import pad_image, pad_bbox, pad_poly, pad_point
from .crop import crop_image, crop_mask
from .resize import resize_image, resize_mask


__all__ = ['MixUp', 'CutMix', 'Mosaic']


@INTERNODE.register_module()
class MixUp(Bamboo):
    def calc_intl_param_forward(self, data_dict):
        index_mix = random.randint(0, data_dict['len_data_lines'] - 1)
        while index_mix == data_dict['index']:
            index_mix = random.randint(0, data_dict['len_data_lines'] - 1)
        data_dict['intl_index_mix'] = index_mix
        data_dict['intl_lam'] = np.random.beta(1.5, 1.5)
        return data_dict

    def erase_intl_param_forward(self, data_dict):
        data_dict.pop('intl_index_mix')
        data_dict.pop('intl_lam')
        return data_dict

    def forward(self, data_dict):
        index_mix = data_dict['intl_index_mix']

        reader = data_dict['reader']
        len_data_lines = data_dict['len_data_lines']

        a = super(MixUp, self).forward(data_dict)

        k = a.keys()
        assert 'mask' not in k and 'point' not in  k

        data_dict['index'] = index_mix
        data_dict['len_data_lines'] = len_data_lines
        data_dict['reader'] = reader
        b = super(MixUp, self).forward(data_dict)

        w_a, h_a = get_image_size(a['image'])
        w_b, h_b = get_image_size(b['image'])
        max_w = max(w_a, w_b)
        max_h = max(h_a, h_b)

        a['image'] = pad_image(a['image'], (0, 0, max(max_w - w_a, 0), max(max_h - h_a, 0)))
        b['image'] = pad_image(b['image'], (0, 0, max(max_w - w_b, 0), max(max_h - h_b, 0)))

        a['ori_size'] = np.array([max_h, max_w]).astype(np.float32)

        lam = data_dict['intl_lam']

        if is_pil(a['image']):
            a['image'] = Image.blend(a['image'], b['image'], 1 - lam)
        else:
            a['image'] = cv2.addWeighted(a['image'], lam, b['image'], 1 - lam, 0)

        if 'bbox' in k:
            a['bbox'] = np.concatenate([a['bbox'], b['bbox']])

            if 'bbox_meta' in k:
                a['bbox_meta']['score'] *= lam
                b['bbox_meta']['score'] *= 1 - lam
                a['bbox_meta'] += b['bbox_meta']

        if 'label' in k:
            for i in range(len(a['label'])):
                a['label'][i] = a['label'][i] * lam + b['label'][i] * (1 - lam)

        if 'path' in k:
            a['path'] = '[mixup]({}, {}, {})'.format(a['path'], b['path'], lam)

        return a

    def rper(self):
        return type(self).__name__ + '(not available)'


@INTERNODE.register_module()
class CutMix(Bamboo):
    def calc_intl_param_forward(self, data_dict):
        index_mix = random.randint(0, data_dict['len_data_lines'] - 1)
        while index_mix == data_dict['index']:
            index_mix = random.randint(0, data_dict['len_data_lines'] - 1)
        data_dict['intl_index_mix'] = index_mix
        data_dict['intl_lam'] = np.random.beta(1.5, 1.5)
        return data_dict

    def erase_intl_param_forward(self, data_dict):
        data_dict.pop('intl_index_mix')
        data_dict.pop('intl_lam')
        return data_dict

    def forward(self, data_dict):
        index_mix = data_dict['intl_index_mix']

        reader = data_dict['reader']
        len_data_lines = data_dict['len_data_lines']

        a = super(CutMix, self).forward(data_dict)

        k = a.keys()
        assert 'bbox' not in k and 'point' not in k and 'poly' not in k

        data_dict['index'] = index_mix
        data_dict['len_data_lines'] = len_data_lines
        data_dict['reader'] = reader
        b = super(CutMix, self).forward(data_dict)

        wa, ha = get_image_size(a['image'])
        wb, hb = get_image_size(b['image'])

        lam = data_dict['intl_lam']

        xa1, ya1, xa2, ya2 = self.rand_boxa(get_image_size(a['image']), lam)

        w_bcut = int((xa2 - xa1) * wa / wb)
        h_bcut = int((ya2 - ya1) * ha / hb)

        if w_bcut > wb or h_bcut > hb:
            r = min(wb / w_bcut, hb / h_bcut)

            w_bcut = int(w_bcut * r)
            h_bcut = int(h_bcut * r)

        xb1 = random.randint(0, wb - w_bcut)
        yb1 = random.randint(0, hb - h_bcut)
        xb2 = xb1 + w_bcut
        yb2 = yb1 + h_bcut

        b_cut = crop_image(b['image'], xb1, yb1, xb2, yb2)
        b_cut = resize_image(b_cut, (xa2 - xa1, ya2 - ya1))

        if is_pil(b['image']):
            a['image'].paste(b_cut, (xa1, ya1, xa2, ya2))
        else:
            a['image'][ya1:ya2, xa1:xa2] = b_cut

        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((xa2 - xa1) * (ya2 - ya1) / (wa * ha))

        if 'path' in k:
            a['path'] = '[cutmix]({}, {}, {})'.format(a['path'], b['path'], lam)

        if 'label' in k:
            for i in range(len(a['label'])):
                a['label'][i] = a['label'][i] * lam + b['label'][i] * (1 - lam)

        if 'mask' in k:
            bmask_cut = crop_mask(b['mask'], xb1, yb1, xb2, yb2)
            bmask_cut = resize_mask(bmask_cut, (xa2 - xa1, ya2 - ya1))
            a['mask'][ya1:ya2, xa1:xa2] = bmask_cut

            return a

    @staticmethod
    def rand_boxa(size, lam):
        W, H = size
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        return x1, y1, x2, y2

    def rper(self):
        return type(self).__name__ + '(not available)'


@INTERNODE.register_module()
class Mosaic(Bamboo):
    def calc_intl_param_forward(self, data_dict):
        data_dict['intl_mosaic_ids'] = [random.randint(0, data_dict['len_data_lines'] - 1) for _ in range(3)]
        return data_dict

    def erase_intl_param_forward(self, data_dict):
        data_dict.pop('intl_mosaic_ids')
        return data_dict

    def forward(self, data_dict):
        indices = data_dict['intl_mosaic_ids']
        
        reader = data_dict['reader']
        len_data_lines = data_dict['len_data_lines']

        d1 = super(Mosaic, self).forward(data_dict)

        k = d1.keys()
        assert 'label' not in k

        data_dict['index'] = indices[0]
        data_dict['len_data_lines'] = len_data_lines
        data_dict['reader'] = reader
        d2 = super(Mosaic, self).forward(data_dict)

        data_dict['index'] = indices[1]
        data_dict['len_data_lines'] = len_data_lines
        data_dict['reader'] = reader
        d3 = super(Mosaic, self).forward(data_dict)

        data_dict['index'] = indices[2]
        data_dict['len_data_lines'] = len_data_lines
        data_dict['reader'] = reader
        d4 = super(Mosaic, self).forward(data_dict)

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

            img4[yc - h1:yc, xc - w1:xc] = d1['image']
            img4[yc - h2:yc, xc:xc + w2] = d2['image']
            img4[yc:yc + h3, xc - w3:xc] = d3['image']
            img4[yc:yc + h4, xc:xc + w4] = d4['image']

        d1['image'] = img4
        d1['ori_size'] = np.array([new_h, new_w]).astype(np.float32)
        d1['path'] = '[mosaic]({}, {}, {}, {})'.format(d1['path'], d2['path'], d3['path'], d4['path'])

        if 'bbox' in k:
            b1 = pad_bbox(d1['bbox'], xc - w1, yc - h1)
            b2 = pad_bbox(d2['bbox'], xc, yc - h2)
            b3 = pad_bbox(d3['bbox'], xc - w3, yc)
            b4 = pad_bbox(d4['bbox'], xc, yc)
            d1['bbox'] = np.concatenate((b1, b2, b3, b4))

            if 'bbox_meta' in k:
                if 'box2point' in d1['bbox_meta'].keys():
                    d2['bbox_meta']['box2point'] += len(d1['point'])
                    d3['bbox_meta']['box2point'] += len(d1['point']) + len(d2['point'])
                    d4['bbox_meta']['box2point'] += len(d1['point']) + len(d2['point']) + len(d3['point'])

                d1['bbox_meta'] += d2['bbox_meta']
                d1['bbox_meta'] += d3['bbox_meta']
                d1['bbox_meta'] += d4['bbox_meta']
                # res['bbox_meta'] = deepcopy(d1['bbox_meta'])

        if 'point' in k:
            p1 = pad_point(d1['point'], xc - w1, yc - h1)
            p2 = pad_point(d2['point'], xc, yc - h2)
            p3 = pad_point(d3['point'], xc - w3, yc)
            p4 = pad_point(d4['point'], xc, yc)
            d1['point'] = np.concatenate((p1, p2, p3, p4))

            if 'point_meta' in k:
                d1['point_meta'] += d2['point_meta']
                d1['point_meta'] += d3['point_meta']
                d1['point_meta'] += d4['point_meta']
                # res['point_meta'] = deepcopy(d1['point_meta'])

        if 'mask' in k:
            mask4 = np.zeros((new_h, new_w)).astype(np.int32)

            mask4[yc - h1:yc, xc - w1:xc] = d1['mask']
            mask4[yc - h2:yc, xc:xc + w2] = d2['mask']
            mask4[yc:yc + h3, xc - w3:xc] = d3['mask']
            mask4[yc:yc + h4, xc:xc + w4] = d4['mask']

            d1['mask'] = mask4

        if 'poly' in k:
            p1 = pad_poly(d1['poly'], xc - w1, yc - h1)
            p2 = pad_poly(d2['poly'], xc, yc - h2)
            p3 = pad_poly(d3['poly'], xc - w3, yc)
            p4 = pad_poly(d4['poly'], xc, yc)
            d1['poly'] = p1 + p2 + p3 + p4

            if 'poly_meta' in k:
                d1['poly_meta'] += d2['poly_meta']
                d1['poly_meta'] += d3['poly_meta']
                d1['poly_meta'] += d4['poly_meta']
                # res['poly_meta'] = deepcopy(d1['poly_meta'])

        return d1

    def rper(self):
        return type(self).__name__ + '(not available)'

