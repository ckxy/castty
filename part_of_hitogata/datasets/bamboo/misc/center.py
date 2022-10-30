import math
import torch
import numpy as np
from ..builder import INTERNODE
from ..base_internode import BaseInternode
from ...utils.common import get_image_size
from ....utils.point_tools import heatmaps2points
from ....utils.heatmap_tools import calc_gaussian_2d, gaussian_radius, gen_gaussian_target
from torch.nn.functional import affine_grid, grid_sample, pad, interpolate


__all__ = ['CalcHeatmapByPoint', 'CalcCenterNetGrids']


@INTERNODE.register_module()
class CalcHeatmapByPoint(BaseInternode):
    def __init__(self, ratio=1, sigma=1, resample=False):
        self.ratio = ratio
        self.sigma = sigma
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

        if not vis or not (0 <= cx < w) or not (0 <= cy < h):
            return img, False

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

        return img, True

    def __call__(self, data_dict):
        w, h = get_image_size(data_dict['image'])
        heatmaps_per_img = []
        visible_per_img = []

        print(data_dict['point'].shape)

        if 'point_meta' in data_dict.keys():
            visible = data_dict['point_meta']['visible']
        else:
            visible = np.empty(shape=data_dict['point'].shape[:2], dtype=np.bool)
            visible.fill(True)

        for points, vises in zip(data_dict['point'], visible):
            for point, vis in zip(points, vises):
                # print('a', vis)
                heatmap, new_vis = self.calc_heatmap((w, h), point, vis)
                # print(new_vis)
                heatmaps_per_img.append(heatmap.unsqueeze(0))
                visible_per_img.append(new_vis)

        visible_per_img = np.array(visible_per_img).reshape(visible.shape)
        data_dict['heatmap'] = torch.cat(heatmaps_per_img, dim=0)

        if 'point_meta' in data_dict.keys():
            data_dict['point_meta']['visible'] = visible_per_img

        return data_dict

    def reverse(self, **kwargs):
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
        return kwargs

    def __repr__(self):
        return 'CalcHeatmapByPoint(ratio={}, sigma={})'.format(self.ratio, self.sigma)


@INTERNODE.register_module()
class CalcCenterNetGrids(BaseInternode):
    def __init__(self, ratio=1, num_classes=1, use_bbox=True, use_point=False, **kwargs):
        self.ratio = ratio
        self.num_classes = num_classes
        self.use_bbox = use_bbox
        self.use_point = use_point

    def __call__(self, data_dict):
        w, h = get_image_size(data_dict['image'])

        center_heatmap_target = torch.zeros(self.num_classes, int(h * self.ratio), int(w * self.ratio)).type(torch.float32)

        gt_bbox = torch.from_numpy(data_dict['bbox'])
        gt_label = torch.from_numpy(data_dict['bbox_meta']['class_id'])
        center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) / 2
        center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) / 2
        gt_centers = torch.cat((center_x, center_y), dim=1) * self.ratio

        if self.use_bbox:
            wh_target = torch.zeros(2, int(h * self.ratio), int(w * self.ratio)).type(torch.float32)
            wh_target_weight = torch.zeros(2, int(h * self.ratio), int(w * self.ratio)).type(torch.float32)

            offset_target = torch.zeros(2, int(h * self.ratio), int(w * self.ratio)).type(torch.float32)
            offset_target_weight = torch.zeros(2, int(h * self.ratio), int(w * self.ratio)).type(torch.float32)

        if self.use_point:
            _, n_points, _ = data_dict['point'].shape

            kpt_heatmap_target = torch.zeros(n_points, int(h * self.ratio), int(w * self.ratio)).type(torch.float32)

            loc_target = torch.zeros(2 * n_points, int(h * self.ratio), int(w * self.ratio)).type(torch.float32)
            loc_target_weight = torch.zeros(2 * n_points, int(h * self.ratio), int(w * self.ratio)).type(torch.float32)

            offset_target = torch.zeros(2, int(h * self.ratio), int(w * self.ratio)).type(torch.float32)
            offset_target_weight = torch.zeros(2, int(h * self.ratio), int(w * self.ratio)).type(torch.float32)

            gt_keypoints = torch.from_numpy(data_dict['point']) * self.ratio
            visible = data_dict['point_meta']['visible']

        for j, ct in enumerate(gt_centers):
            ctx_int, cty_int = ct.int()
            ctx, cty = ct
            scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * self.ratio
            scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * self.ratio
            radius = gaussian_radius([scale_box_h, scale_box_w], min_overlap=0.3)
            radius = max(0, int(radius))
            ind = gt_label[j]
            gen_gaussian_target(center_heatmap_target[ind], [ctx_int, cty_int], radius)

            if self.use_bbox:
                wh_target[0, cty_int, ctx_int] = scale_box_w
                wh_target[1, cty_int, ctx_int] = scale_box_h

                offset_target[0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[1, cty_int, ctx_int] = cty - cty_int

                wh_target_weight[:, cty_int, ctx_int] = 1
                offset_target_weight[:, cty_int, ctx_int] = 1

            if self.use_point:
                for k, kpt in enumerate(gt_keypoints[j]):
                    if not visible[j][k]:
                        continue

                    kpx_int, kpy_int = kpt.int()
                    kpx, kpy = kpt
                    gen_gaussian_target(kpt_heatmap_target[k], [kpx_int, kpy_int], radius)

                    loc_target[2 * k, kpy_int, kpx_int] = kpx - ctx
                    loc_target[2 * k + 1, kpy_int, kpx_int] = kpy - cty

                    loc_target_weight[2 * k, kpy_int, kpx_int] = 1
                    loc_target_weight[2 * k + 1, kpy_int, kpx_int] = 1

                    offset_target[0, kpy_int, kpx_int] = kpx - kpx_int
                    offset_target[1, kpy_int, kpx_int] = kpy - kpy_int

                    offset_target_weight[:, kpy_int, kpx_int] = 1
                    # print(loc_target.shape)
                    # print(k, kpx_int, kpy_int, visible[j][k])
                    # exit()

        data_dict['center_heatmap'] = center_heatmap_target

        if self.use_bbox:
            data_dict['wh_map'] = wh_target
            data_dict['wh_weight_map'] = wh_target_weight

            data_dict['offset_map'] = offset_target
            data_dict['offset_target_weight'] = offset_target_weight

        if self.use_point:
            data_dict['kpt_heatmap'] = kpt_heatmap_target

            data_dict['loc_map'] = loc_target
            data_dict['loc_weight_map'] = loc_target_weight

            data_dict['offset_map'] = offset_target
            data_dict['offset_target_weight'] = offset_target_weight
        return data_dict

    def __repr__(self):
        return 'CalcCenterNetGrids(ratio={}, num_classes={}, use_bbox={}, use_point={})'.format(self.ratio, self.num_classes, self.use_bbox, self.use_point)

