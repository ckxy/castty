import cv2
import torch
import numpy as np
from ..base_internode import BaseInternode
from torch.nn.functional import interpolate
from utils.heatmap_tools import calc_gaussian_2d, heatmap2quad


__all__ = ['CalcAffinityQuad', 'CalcHeatmapByQuad', 'RandomCropSequence']


class CalcAffinityQuad(BaseInternode):
    def __call__(self, data_dict):
        n = len(data_dict['quad'])

        if n < 2:
            data_dict['quad_affinity'] = np.zeros([0, 4, 2], dtype=np.float32)
        else:
            data_dict['quad_affinity'] = []

            for i in range(1, n):
                bbox_1 = data_dict['quad'][i - 1]
                bbox_2 = data_dict['quad'][i]

                tl, bl = self.get_centers_of_triangles(bbox_1)
                tr, br = self.get_centers_of_triangles(bbox_2)

                affinity = np.array([tl, tr, br, bl])
                data_dict['quad_affinity'].append(affinity)

            data_dict['quad_affinity'] = np.array(data_dict['quad_affinity'], dtype=np.float32)

        return data_dict

    def get_centers_of_triangles(self, quad):
        u = np.min(quad[..., 1])
        # print(quad, u)

        centers_of_triangles = []
        center = np.mean(quad, axis=0)
        # print(center)
        for i in range(len(quad)):
            p1 = quad[i]
            p2 = quad[(i + 1) % len(quad)]
            centers_of_triangles.append(np.mean([p1, p2, center], axis=0))
            # print(p1, p2, np.mean([p1, p2, center], axis=0))
        centers_of_triangles = np.array(centers_of_triangles)
        # print(centers_of_triangles)
        top = np.argmin(centers_of_triangles[..., 1])
        bottom = np.argmax(centers_of_triangles[..., 1])
        # print(top, bottom)
        # exit()
        return centers_of_triangles[top], centers_of_triangles[bottom]


    # def reverse(self, **kwargs):
    #     if 'quad_affinity' in kwargs.keys():
    #         kwargs['quad'] = kwargs.pop('quad_affinity')
    #         kwargs['have_affinity'] = True
    #     if 'heatmap_affinity' in kwargs.keys():
    #         kwargs['heatmap'] = kwargs.pop('heatmap_affinity')
    #         kwargs['have_affinity'] = True
    #     return kwargs

    def __repr__(self):
        return 'CalcAffinityQuad()'

    def rper(self):
        return 'CalcAffinityQuad()'


class CalcHeatmapByQuad(BaseInternode):
    def __init__(self, thresh, ratio=1):
        self.ratio = ratio

        self.gaussian = calc_gaussian_2d(alpha=False)
        self.threshold = thresh
        self.box = self.get_box()

    def get_box(self):
        h, w = self.gaussian.shape
        binary = self.gaussian >= self.threshold
        x1 = np.min(np.nonzero(binary)[1])
        y1 = np.min(np.nonzero(binary)[0])
        x2 = min(np.max(np.nonzero(binary)[1]) + 1, w)
        y2 = min(np.max(np.nonzero(binary)[0]) + 1, h)
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

    def calc_heatmap(self, h, w, quad):
        m = cv2.getPerspectiveTransform(self.box, quad / self.ratio)
        dst = cv2.warpPerspective(self.gaussian, m, (int(w / self.ratio), int(h / self.ratio)), borderValue=0, borderMode=cv2.BORDER_CONSTANT)
        return dst

    def __call__(self, data_dict):
        assert data_dict['quad'].shape[1] == 4

        _, h, w = data_dict['image'].shape

        heatmap = np.zeros((int(h / self.ratio), int(w / self.ratio)), dtype=np.float32)
        for quad in data_dict['quad']:
            heatmap_tmp = self.calc_heatmap(h, w, quad)
            heatmap = np.where(heatmap_tmp > heatmap, heatmap_tmp, heatmap)

        data_dict['heatmap'] = torch.from_numpy(heatmap)

        return data_dict

    def reverse(self, **kwargs):
        if 'chbq' in kwargs and kwargs['chbq']:
            return kwargs

        if 'training' in kwargs.keys() and kwargs['training']:
            if 'heatmap' in kwargs.keys():
                if kwargs['heatmap'].dim() == 3:
                    kwargs['heatmap'] = kwargs['heatmap'].unsqueeze(0)
                    _, n, h, w = kwargs['heatmap'].shape
                    kwargs['heatmap'] = interpolate(kwargs['heatmap'], size=(int(h * self.ratio), int(w * self.ratio)), mode='bilinear', align_corners=False)
                    kwargs['heatmap'] = kwargs['heatmap'][0]
                else:
                    # print(kwargs['heatmap'].shape)
                    kwargs['heatmap'] = kwargs['heatmap'].unsqueeze(0).unsqueeze(0)
                    _, n, h, w = kwargs['heatmap'].shape
                    kwargs['heatmap'] = interpolate(kwargs['heatmap'], size=(int(h * self.ratio), int(w * self.ratio)), mode='bilinear', align_corners=False)
                    kwargs['heatmap'] = kwargs['heatmap'][0][0]
                    # print(kwargs['heatmap'].shape,'=============')
                kwargs['chbq'] = True
        else:
            if 'heatmap' in kwargs.keys():
                heatmap = kwargs.pop('heatmap').detach().cpu().numpy()
                quad = heatmap2quad(heatmap, 1)
                kwargs['quad'] = quad * self.ratio

                kwargs['chbq'] = True

        return kwargs

    def __repr__(self):
        return 'CalcHeatmapByQuad(ratio={}, thresh={})'.format(self.ratio, self.threshold)

    def rper(self):
        return 'CalcHeatmapByQuad(ratio={})'.format(1 / self.ratio)


class RandomCropSequence(BaseInternode):
    def __init__(self, p=1):
        assert 0 < p <= 1
        self.p = p

    def __call__(self, data_dict):
        # assert 'quad_group' not in data_dict.keys()
        assert isinstance(data_dict['quad'], np.ndarray)

        if random.random() < self.p and len(data_dict['quad']) > 1:
            length = random.randint(1, len(data_dict['quad']))
            left = random.randint(0, len(data_dict['quad']) - length)
            data_dict['quad'] = data_dict['quad'][left:left + length]

            tmp = data_dict['quad'].reshape(-1, 2)
            xmin = np.min(tmp[:, 0])
            xmax = np.max(tmp[:, 0])
            ymin = np.min(tmp[:, 1])
            ymax = np.max(tmp[:, 1])

            data_dict['quad'][..., 0] -= xmin
            data_dict['quad'][..., 1] -= ymin

            data_dict['image'] = data_dict['image'].crop((xmin, ymin, xmax, ymax))

            # print(data_dict['seq'])
            if 'seq' in data_dict.keys():
                data_dict['seq'] = data_dict['seq'][left:left + length]
            # print(data_dict['quad'].shape, length, left, len(data_dict['seq']))
        # exit()

        return data_dict

    def reverse(self, **kwargs):
        kwargs['jump'] = 'all'
        return kwargs

    def __repr__(self):
        return 'RandomCropSequence(p={})'.format(self.p)

    def rper(self):
        return 'RandomCropSequence(not available)'