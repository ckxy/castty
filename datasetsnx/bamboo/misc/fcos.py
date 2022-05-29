import math
import torch
import numpy as np
from ..base_internode import BaseInternode
from utils.bbox_tools import get_xy_grid2


__all__ = ['CalcFCOSGrids']


def calc_centerness(offsets):
    left_right = offsets[:, [0, 2]]
    top_bottom = offsets[:, [1, 3]]
    centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                  (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(centerness)


class CalcFCOSGrids(BaseInternode):
    def __init__(self, strides, multiples=(4, 8), center_sampling_radius=0, img_size=None, to_remove=False, norm=False, analysis=False):
        assert len(multiples) == 2 and multiples[0] < multiples[1]

        self.multiples = multiples
        self.strides = strides
        self.center_sampling_radius = center_sampling_radius
        self.grid_analysis = analysis

        # self.INF = 134217728
        self.INF = 100000000
        self.boxes_sizes = []
        for stride in self.strides:
            self.boxes_sizes.append([stride * self.multiples[0], stride * self.multiples[1]])
        self.boxes_sizes[0][0] = -1
        self.boxes_sizes[-1][-1] = self.INF

        if img_size is None:
            self.use_fixes_size = False
            self.img_size = None
        else:
            grids = []
            limits = []
            h, w = img_size

            for i, stride in enumerate(self.strides):
                grid = ((get_xy_grid2(math.ceil(w / stride), math.ceil(h / stride), 1) + 0.5) * stride).squeeze().reshape(-1, 2)
                grids.append(grid)

                limit = torch.ones(math.ceil(h / stride) * math.ceil(w / stride), 2)
                limit[:, 0] *= self.boxes_sizes[i][0]
                limit[:, 1] *= self.boxes_sizes[i][1]
                limits.append(limit)
            self.grids = torch.cat(grids, dim=0)
            self.limits = torch.cat(limits, dim=0)

            self.use_fixes_size = True
            self.img_size = img_size

        if to_remove:
            self.to_remove = 1
        else:
            self.to_remove = 0

        self.norm = norm

    def __call__(self, data_dict):
        _, h, w = data_dict['image'].shape

        grids_per_level = [math.ceil(h / s) * math.ceil(w / s) for s in self.strides]

        offset, label, centerness, mixup, ga_id = self.calc_targets(h, w, data_dict['bbox'])

        data_dict['gt_offset'] = torch.split(offset, grids_per_level, dim=0)
        data_dict['gt_label'] = torch.split(label, grids_per_level, dim=0)
        data_dict['gt_centerness'] = torch.split(centerness, grids_per_level, dim=0)
        data_dict['gt_mixup'] = torch.split(mixup, grids_per_level, dim=0)

        if self.norm:
            data_dict['gt_offset'] = list(data_dict['gt_offset'])
            for i in range(len(self.strides)):
                data_dict['gt_offset'][i] = data_dict['gt_offset'][i] / self.strides[i]
            data_dict['gt_offset'] = tuple(data_dict['gt_offset'])

        # print(offset.shape, label.shape, centerness.shape, mixup.shape)
        # print([a.shape for a in data_dict['gt_offset']])
        # exit()

        if ga_id is not None:
            nc = 0
            bg = []
            ig = []
            for k, stride in enumerate(self.strides):
                bg.append([])
                ig.append([])

                gh, gw = math.ceil(h / stride), math.ceil(w / stride)
                mask = (ga_id[..., 0] >= nc) & (ga_id[..., 0] < nc + gh * gw)

                if ga_id[mask].shape[0] > 0:
                    for gid, bi in ga_id[mask]:
                        gid -= nc
                        ig[k].append([int(gid - gw * math.floor(gid / gw)), math.floor(gid / gw), int(bi)])

                    bis = np.unique(ga_id[mask][..., 1])
                    for bi in bis:
                        bg[k].append(data_dict['bbox'][:, :4][int(bi)].astype(np.int).tolist() + [int(bi)])
                nc += gh * gw

            data_dict['ga_bbox'] = bg
            data_dict['ga_index'] = ig

        return data_dict

    def calc_targets(self, h, w, boxes):
        if self.use_fixes_size:
            grids = self.grids
            limits = self.limits
        else:
            grids = []
            limits = []
            for i, stride in enumerate(self.strides):
                grid = ((get_xy_grid2(math.ceil(w / stride), math.ceil(h / stride), 1) + 0.5) * stride).squeeze().reshape(-1, 2)
                grids.append(grid)

                limit = torch.ones(math.ceil(h / stride) * math.ceil(w / stride), 2)
                limit[:, 0] *= self.boxes_sizes[i][0]
                limit[:, 1] *= self.boxes_sizes[i][1]
                limits.append(limit)
            grids = torch.cat(grids, dim=0)
            limits = torch.cat(limits, dim=0)

        target = torch.from_numpy(boxes)

        cx, cy = grids[:, 0:1], grids[:, 1:2]

        label = target[:, 4]
        mixup = target[:, 5]

        l = cx - target[:, 0:1].T
        t = cy - target[:, 1:2].T
        r = target[:, 2:3].T - cx
        b = target[:, 3:4].T - cy
        offset_target = torch.stack([l, t, r, b], dim=2)

        if self.center_sampling_radius > 0:
            raise ValueError
        else:
            in_ind = offset_target.min(dim=2)[0] > 0

        max_offset = offset_target.max(dim=2)[0]

        wi_lim_ind = (max_offset >= limits[:, 0:1]) & (max_offset <= limits[:, 1:2])

        area = (target[..., 2] - target[..., 0] + self.to_remove) * (target[..., 3] - target[..., 1] + self.to_remove)

        area = area[None].repeat(len(grids), 1)
        area[in_ind == 0] = self.INF
        area[wi_lim_ind == 0] = self.INF

        min_area, min_area_id = area.min(dim=1)

        offset_target = offset_target[range(len(grids)), min_area_id]

        centerness_target = calc_centerness(offset_target)

        label_target = label[min_area_id]
        label_target[min_area == self.INF] = 0

        mixup_target = mixup[min_area_id]
        mixup_target[min_area == self.INF] = 0

        # a = torch.load('/home/ubuntu/test/fcos/FCOS/p.pth')
        # print(label_target)
        # print(a.shape, label_target.shape)
        # print((a == label_target).all())
        # exit()

        if self.grid_analysis:
            ga_id = torch.nonzero(in_ind & wi_lim_ind)
        else:
            ga_id = None

        return offset_target, label_target, centerness_target, mixup_target, ga_id

    def __repr__(self):
        return 'CalcFCOSGrids(strides={}, multiples={}, img_size={}, center_sampling_radius={}, norm={}, analysis={})'.format(self.strides, self.multiples, self.img_size, self.center_sampling_radius, self.norm, self.grid_analysis)

    def rper(self):
        return 'CalcFCOSGrids(not available)'
