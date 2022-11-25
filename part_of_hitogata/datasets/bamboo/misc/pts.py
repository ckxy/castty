import math
import torch
import numpy as np
from ..builder import INTERNODE
from ..base_internode import BaseInternode
from ...utils.common import get_image_size
from torch.nn.functional import pairwise_distance
from ....utils.heatmap_tools import calc_gaussian_2d, gaussian_radius, gen_gaussian_target


__all__ = ['CalcPTSGrids']


@INTERNODE.register_module()
class CalcPTSGrids(BaseInternode):
    def __init__(self, ratio=1, **kwargs):
        self.ratio = ratio

    def calc_kpt2cen(self, kpt, table_id, cens, all_table_ids):
        kpx, kpy = kpt
        centers = cens[all_table_ids == table_id]
        dis = pairwise_distance(centers, kpt)

        lt = (centers[..., 0] < kpx) & (centers[..., 1] < kpy)
        rt = (centers[..., 0] >= kpx) & (centers[..., 1] < kpy)
        rb = (centers[..., 0] >= kpx) & (centers[..., 1] >= kpy)
        lb = (centers[..., 0] < kpx) & (centers[..., 1] >= kpy)

        res = torch.zeros(8, dtype=torch.float32)
        res_weights = torch.zeros(8, dtype=torch.float32)

        self.get_value(kpt, centers[lt], dis[lt], res, res_weights, 0)
        self.get_value(kpt, centers[rt], dis[rt], res, res_weights, 1)
        self.get_value(kpt, centers[rb], dis[rb], res, res_weights, 2)
        self.get_value(kpt, centers[lb], dis[lb], res, res_weights, 3)

        return res, res_weights

    def get_value(self, kpt, centers, dis, res, res_weights, k):
        if len(dis) == 0:
            return
        min_id = torch.argmin(dis)
        ctx, cty = centers[min_id]
        kpx, kpy = kpt

        res[2 * k] = ctx - kpx
        res[2 * k + 1] = cty - kpy
        res_weights[2 * k] = 1
        res_weights[2 * k + 1] = 1

    def forward(self, data_dict):
        w, h = get_image_size(data_dict['image'])

        center_heatmap_target = torch.zeros(1, int(h * self.ratio), int(w * self.ratio)).type(torch.float32)
        kpt_heatmap_target = torch.zeros(1, int(h * self.ratio), int(w * self.ratio)).type(torch.float32)

        cen2kpt_target = torch.zeros(8, int(h * self.ratio), int(w * self.ratio)).type(torch.float32)
        cen2kpt_target_weight = torch.zeros(8, int(h * self.ratio), int(w * self.ratio)).type(torch.float32)

        kpt2cen_target = torch.zeros(8, int(h * self.ratio), int(w * self.ratio)).type(torch.float32)
        kpt2cen_target_weight = torch.zeros(8, int(h * self.ratio), int(w * self.ratio)).type(torch.float32)

        cen_offset_target = torch.zeros(2, int(h * self.ratio), int(w * self.ratio)).type(torch.float32)
        cen_offset_target_weight = torch.zeros(2, int(h * self.ratio), int(w * self.ratio)).type(torch.float32)

        kpt_offset_target = torch.zeros(2, int(h * self.ratio), int(w * self.ratio)).type(torch.float32)
        kpt_offset_target_weight = torch.zeros(2, int(h * self.ratio), int(w * self.ratio)).type(torch.float32)

        gt_bbox = torch.from_numpy(data_dict['bbox'])
        # gt_label = torch.from_numpy(data_dict['bbox_meta']['class_id'])
        center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) / 2
        center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) / 2
        gt_centers = torch.cat((center_x, center_y), dim=1) * self.ratio
        center_meta = data_dict['bbox_meta']

        gt_keypoints = torch.from_numpy(data_dict['point']) * self.ratio
        visible = data_dict['point_meta']['visible']

        for j, ct in enumerate(gt_centers):
            ctx_int, cty_int = ct.int()
            ctx, cty = ct
            scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * self.ratio
            scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * self.ratio
            radius = gaussian_radius([scale_box_h, scale_box_w], min_overlap=0.3)
            radius = max(0, int(radius))
            # ind = gt_label[j]
            gen_gaussian_target(center_heatmap_target[0], [ctx_int, cty_int], radius)

            for i in range(4):
                if not visible[j][i]:
                    continue
                # 到底是浮点还是取整？
                kpx, kpy = gt_keypoints[j][i]
                # 跟论文里图3下面公式1的计算方法不一样
                cen2kpt_target[2 * i, cty_int, ctx_int] = kpx - ctx
                cen2kpt_target[2 * i + 1, cty_int, ctx_int] = kpy - cty
                cen2kpt_target_weight[2 * i, cty_int, ctx_int] = 1
                cen2kpt_target_weight[2 * i + 1, cty_int, ctx_int] = 1

            cen_offset_target[0, cty_int, ctx_int] = ctx - ctx_int
            cen_offset_target[1, cty_int, ctx_int] = cty - cty_int
            cen_offset_target_weight[:, cty_int, ctx_int] = 1

            # table_id = center_meta['table_id'][j]
            # startcol = center_meta['startcol'][j]
            # endcol = center_meta['endcol'][j]
            # startrow = center_meta['startrow'][j]
            # endrow = center_meta['endrow'][j]
            # print(j, table_id, startcol, endcol, startrow, endrow)
            # exit()

            for k, kpt in enumerate(gt_keypoints[j]):
                if not visible[j][k]:
                    continue

                kpx_int, kpy_int = kpt.int()
                kpx, kpy = kpt
                # print(j, k, kpt)
                gen_gaussian_target(kpt_heatmap_target[0], [kpx_int, kpy_int], radius)

                res, res_weights = self.calc_kpt2cen(kpt, center_meta['table_id'][j], gt_centers, center_meta['table_id'])

                kpt2cen_target[:, kpy_int, kpx_int] = res
                kpt2cen_target_weight[:, kpy_int, kpx_int] = res_weights

                kpt_offset_target[0, kpy_int, kpx_int] = kpx - kpx_int
                kpt_offset_target[1, kpy_int, kpx_int] = kpy - kpy_int
                kpt_offset_target_weight[:, kpy_int, kpx_int] = 1

        data_dict['center_heatmap'] = center_heatmap_target
        data_dict['kpt_heatmap'] = kpt_heatmap_target

        data_dict['center_offset_map'] = cen_offset_target
        data_dict['center_offset_target_weight'] = cen_offset_target_weight

        data_dict['cen2kpt_map'] = cen2kpt_target
        data_dict['cen2kpt_weight_map'] = cen2kpt_target_weight

        data_dict['kpt2cen_map'] = kpt2cen_target
        data_dict['kpt2cen_weight_map'] = kpt2cen_target_weight

        data_dict['keypoint_offset_map'] = kpt_offset_target
        data_dict['keypoint_offset_target_weight'] = kpt_offset_target_weight

        return data_dict

    def __repr__(self):
        return 'CalcPTSGrids(ratio={})'.format(self.ratio)

