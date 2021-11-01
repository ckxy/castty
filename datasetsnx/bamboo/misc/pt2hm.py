import math
import torch
import numpy as np
from ..base_internode import BaseInternode
from utils.point_tools import heatmaps2points
from utils.heatmap_tools import calc_gaussian_2d
from torch.nn.functional import affine_grid, grid_sample, pad, interpolate


__all__ = ['CalcHeatmapByPoint']


class CalcHeatmapByPoint(BaseInternode):
    def __init__(self, ratio=1, sigma=1, no_forward=False, resample=False):
        self.ratio = ratio
        self.sigma = sigma
        self.no_forward = no_forward
        self.resample = resample

        self.w = torch.from_numpy(calc_gaussian_2d(sigma, step=1, alpha=False))
        if resample:
            p = 1
            self.w = pad(self.w, (p, p, p, p), 'constant', 0)
            self.w = self.w.unsqueeze(0).unsqueeze(0)
            self.a = 2 / (6 * self.sigma + 1 + 2 * p)
        # print(self.w.numpy(), self.w.numpy().shape)
        # exit()

    def calc_heatmap(self, size, pt, vis):
        w, h = size
        w = int(self.ratio * w)
        h = int(self.ratio * h)
        x, y = self.ratio * pt
        cx, cy = round(x), round(y)

        img = torch.zeros(h, w)

        if vis == 0 or ~(0 <= cx < w) or ~(0 <= cy < h):
            return img, 0

        if self.resample:
            offset_x = x - cx
            offset_y = y - cy
            # offset_x = offset_y = 1
            # print(x, y)
            # print(cx, cy)
            # print(offset_x, offset_y)
            # print(self.w.numpy(), self.w.numpy().shape)

            # a = 2 / (2 * radius + 1)
            theta = torch.zeros(1, 2, 3)
            theta[0, 0, 0] = 1
            theta[0, 1, 1] = 1
            theta[0, 0, 2] = offset_x * self.a
            theta[0, 1, 2] = offset_y * self.a
            # print(theta)

            grid = affine_grid(theta, self.w.size(), align_corners=False)
            warp_w = grid_sample(self.w, grid, align_corners=False).squeeze()

            radius = (warp_w.shape[-1] - 1) / 2
            # print(warp_w.numpy(), warp_w.numpy().shape, radius)
            # exit()
        else:
            # print(x, y)
            # print(cx, cy)
            # print(img.shape, img.dtype)

            radius = 3 * self.sigma
            warp_w = self.w

        l = max(0, int(cx - radius))
        t = max(0, int(cy - radius))
        r = min(int(cx + radius) + 1, w)
        b = min(int(cy + radius) + 1, h)

        lo = int(radius - cx) if radius > cx else 0
        to = int(radius - cy) if radius > cy else 0
        ro = int(cx + radius + 1 - w) if cx + radius + 1 > w else 0
        bo = int(cy + radius + 1 - h) if cy + radius + 1 > h else 0

        wh, ww = warp_w.shape

        # print(radius, warp_w.shape)
        # print(l, t, r, b)
        # print(lo, to, ro, bo)
        # print(h - bo, w - ro)
        # print(img[t:b, l:r].shape, warp_w[to:(wh - bo), lo:(ww - ro)].shape)
        img[t:b, l:r] = warp_w[to:(wh - bo), lo:(ww - ro)]
        # exit()

        return img, 1

    def __call__(self, data_dict):
        # print(data_dict['path'])
        # print(data_dict['point'])
        if self.no_forward:
            return data_dict

        _, h, w = data_dict['image'].shape
        heatmaps_per_img = []
        visible_per_img = []

        if 'visible' in data_dict.keys():
            visible = data_dict['visible']
        else:
            visible = [None] * len(data_dict['point'])

        for point, vis in zip(data_dict['point'], visible):
            # print(point)
            heatmap, new_vis = self.calc_heatmap((w, h), point, vis)
            heatmaps_per_img.append(heatmap.unsqueeze(0))
            visible_per_img.append(new_vis)

        data_dict['heatmap'] = torch.cat(heatmaps_per_img, dim=0)
        data_dict['visible'] = np.array(visible_per_img).astype(np.int)

        return data_dict

    def reverse(self, **kwargs):
        if 'training' in kwargs.keys() and kwargs['training']:
            if 'heatmap' in kwargs.keys():
                if kwargs['heatmap'].dim() == 3:
                    kwargs['heatmap'] = kwargs['heatmap'].unsqueeze(0)
                    _, n, h, w = kwargs['heatmap'].shape
                    kwargs['heatmap'] = interpolate(kwargs['heatmap'], size=(int(h / self.ratio), int(w / self.ratio)), mode='bilinear', align_corners=False)
                    kwargs['heatmap'] = kwargs['heatmap'][0]
                else:
                    kwargs['heatmap'] = kwargs['heatmap'].unsqueeze(0).unsqueeze(0)
                    _, n, h, w = kwargs['heatmap'].shape
                    kwargs['heatmap'] = interpolate(kwargs['heatmap'], size=(int(h / self.ratio), int(w / self.ratio)), mode='bilinear', align_corners=False)
                    kwargs['heatmap'] = kwargs['heatmap'][0][0]
        else:
            if 'heatmap' in kwargs.keys():
                kwargs['point'] = heatmaps2points(kwargs['heatmap'].unsqueeze(0))[0][0] / self.ratio
        return kwargs

    def __repr__(self):
        if self.no_forward:
            return 'CalcHeatmapByPoint(not available)'
        else:
            return 'CalcHeatmapByPoint(ratio={}, sigma={})'.format(self.ratio, self.sigma)

    def rper(self):
        return 'CalcHeatmapByPoint(ratio={})'.format(1 / self.ratio)
