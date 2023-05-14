import torch
import numpy as np
from ..builder import INTERNODE
from ..base_internode import BaseInternode
from .psenet import generate_effective_mask, generate_kernel
from ...utils.common import get_image_size
from ....utils.bbox_tools import xyxy2xywh, get_grid2, calc_iou2


__all__ = ['CalcNanoGrids']


@INTERNODE.register_module()
class CalcNanoGrids(BaseInternode):
    def __init__(self, scale, strides, num_classes, top_k=9, analysis=False, **kwargs):
        self.scale = scale
        self.strides = strides
        self.num_classes = num_classes
        self.top_k = top_k
        self.analysis = analysis
        self.INF = 100000000

        if len(self.strides) > 1:
            for i in range(1, len(self.strides)):
                assert self.strides[i] > self.strides[i - 1]

        BaseInternode.__init__(self, **kwargs)

    def forward(self, data_dict, **kwargs):
        # _, h, w = data_dict['image'].shape
        w, h = get_image_size(data_dict['image'])
        featmap_sizes = [(h // s, w // s) for s in self.strides]

        grids = []
        num_grids = []
        for i, (fh, fw) in enumerate(featmap_sizes):
            y, x = get_grid2(fh, fw)
            y, x = y.flatten() * self.strides[i], x.flatten() * self.strides[i]
            cell_size = self.strides[i] * self.scale

            grid = torch.stack(
                [x - 0.5 * cell_size, y - 0.5 * cell_size,
                 x + 0.5 * cell_size, y + 0.5 * cell_size], dim=-1
            )
            num_grids.append(grid.shape[0])
            grids.append(grid)

        grids = torch.cat(grids)
        # print(data_dict['bbox'].shape)
        # print(data_dict['bbox_meta'])

        # bboxes, labels, weights = torch.split(torch.from_numpy(data_dict['bbox']).type(torch.float), (2, 1, 1), dim=1)
        bboxes = torch.from_numpy(data_dict['bbox'])
        labels = torch.from_numpy(data_dict['bbox_meta']['class_id']).unsqueeze(-1).type(torch.float32)
        weights = torch.from_numpy(data_dict['bbox_meta']['score']).unsqueeze(-1).type(torch.float32)
        # print(bboxes.shape, labels.shape, weights.shape)
        # bboxes = bboxes.type(torch.float)
        # labels = labels.type(torch.long)
        # weights = weights.type(torch.float)

        # print(bboxes.shape, labels.shape, weights.shape)
        # print(bboxes)
        # print(labels)
        # print(weights)
        # exit()

        overlaps = calc_iou2(grids.unsqueeze(1), bboxes.unsqueeze(0))

        assigned_gt_inds = overlaps.new_full((grids.shape[0],), 0, dtype=torch.long)
        # assigned_labels = assigned_gt_inds.new_full((grids.shape[0],), -1)

        if bboxes.shape[0] > 0:
            gt_center = xyxy2xywh(bboxes.clone())[:, :2]
            grids_center = xyxy2xywh(grids.clone())[:, :2]

            distances = (grids_center.unsqueeze(1) -
                         gt_center.unsqueeze(0)).pow(2).sum(-1).sqrt()

            distances = torch.split(distances, num_grids, dim=0)
            start_ids = [0] + np.array(num_grids[:-1]).cumsum().tolist()

            candidate_ids = []
            for sid, d in zip(start_ids, distances):
                _, topk_ids_per_level = d.topk(self.top_k, dim=0, largest=False)
                candidate_ids.append(topk_ids_per_level + sid)
            candidate_ids = torch.cat(candidate_ids, dim=0)

            candidate_overlaps = overlaps[candidate_ids, torch.arange(bboxes.shape[0])]
            overlaps_mean_per_gt = candidate_overlaps.mean(0)
            overlaps_std_per_gt = candidate_overlaps.std(0)
            overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

            is_pos = candidate_overlaps >= overlaps_thr_per_gt.unsqueeze(0)

            for gt_idx in range(bboxes.shape[0]):
                candidate_ids[:, gt_idx] += gt_idx * grids.shape[0]
            candidate_ids = candidate_ids.view(-1)

            ep_bboxes_cx = grids_center[:, 0].view(1, -1).expand(bboxes.shape[0], grids.shape[0]).contiguous().view(-1)
            ep_bboxes_cy = grids_center[:, 1].view(1, -1).expand(bboxes.shape[0], grids.shape[0]).contiguous().view(-1)

            l_ = ep_bboxes_cx[candidate_ids].view(-1, bboxes.shape[0]) - bboxes[:, 0]
            t_ = ep_bboxes_cy[candidate_ids].view(-1, bboxes.shape[0]) - bboxes[:, 1]
            r_ = bboxes[:, 2] - ep_bboxes_cx[candidate_ids].view(-1, bboxes.shape[0])
            b_ = bboxes[:, 3] - ep_bboxes_cy[candidate_ids].view(-1, bboxes.shape[0])
            is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
            is_pos = is_pos & is_in_gts

            overlaps_inf = torch.full_like(overlaps, -self.INF).t().contiguous().view(-1)
            index = candidate_ids.view(-1)[is_pos.view(-1)]
            overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
            overlaps_inf = overlaps_inf.view(bboxes.shape[0], -1).t()

            max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
            assigned_gt_inds[max_overlaps != -self.INF] = argmax_overlaps[max_overlaps != -self.INF] + 1  # background 0

            # assigned_labels = assigned_gt_inds.new_full((grids.shape[0],), -1)
            # pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            # if pos_inds.numel() > 0:
            #     gt_labels = boxes[:, 4].type(torch.long)
            #     assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]

        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(-1)
        neg_inds = torch.nonzero(assigned_gt_inds == 0, as_tuple=False).squeeze(-1)
        pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1

        # print(pos_assigned_gt_inds, pos_assigned_gt_inds.shape)

        pos_gt_bboxes = bboxes[pos_assigned_gt_inds, :]
        # print(pos_gt_bboxes.shape)

        # x1, y1, x2, y2, label, weight, bbox_weight, label_weight
        targets = torch.zeros(grids.shape[0], 8).type(torch.float)
        targets[:, 4] = self.num_classes
        targets[:, 7] = 1.0

        targets[pos_inds, :4] = bboxes[pos_assigned_gt_inds, :]
        targets[pos_inds, 4:5] = labels[pos_assigned_gt_inds]
        targets[pos_inds, 5:6] = weights[pos_assigned_gt_inds]
        targets[pos_inds, 6:7] = 1.0

        # if boxes.shape[0] > 0:
        #     pos_gt_bboxes = boxes[:, :4][pos_assigned_gt_inds, :]
        # else:
        #     # hack for index error case
        #     assert pos_assigned_gt_inds.numel() == 0
        #     pos_gt_bboxes = torch.empty_like(boxes).view(-1, 4)

        # bbox_targets = torch.zeros_like(grids)
        # bbox_weights = torch.zeros_like(grids)
        # labels = grids.new_full((grids.shape[0],), self.num_classes, dtype=torch.long)
        # label_weights = grids.new_ones(grids.shape[0], dtype=torch.float)

        # if len(pos_inds) > 0:
        #     bbox_targets[pos_inds, :] = pos_gt_bboxes
        #     bbox_weights[pos_inds, :] = 1.0
        #     gt_labels = boxes[:, 4].type(torch.long)
        #     labels[pos_inds] = gt_labels[pos_assigned_gt_inds]

        # b = targets[:, 4]
        # a = torch.load('/home/ubuntu/test/nanodet/p.pth')
        # print(a.shape, b.shape)
        # print((a == b).all())

        data_dict['nano_target'] = targets
        data_dict['num_pos'] = pos_inds.numel()
        data_dict['num_neg'] = neg_inds.numel()
        data_dict['nano_grid'] = grids
        data_dict['nano_fs'] = featmap_sizes

        if self.analysis:
            data_dict['ga_bbox'] = [[] for _ in range(len(self.strides))]
            data_dict['ga_index'] = [[] for _ in range(len(self.strides))]
            data_dict['nano_pnc'] = [[0, 0] for _ in range(len(self.strides))]

            # print(pos_inds, pos_assigned_gt_inds)
            # print(featmap_sizes)
            start = [s[0] * s[1] for s in featmap_sizes]
            start = [0] + start[:-1]
            for i in range(1, len(start)):
                start[i] += start[i - 1]

            # pos_inds = [100, 1700, 2001]
            pos_bboxes = bboxes[pos_assigned_gt_inds, :]

            for i in range(len(pos_inds)):
                k = 1
                # print(pos_inds[i])
                for j in range(1, len(start)):
                    if pos_inds[i] >= start[j]:
                        k += 1

                data_dict['nano_pnc'][k - 1][0] += 1
                index = pos_inds[i].item() - start[k - 1]
                ii = index // featmap_sizes[k - 1][1]
                jj = index - ii * featmap_sizes[k - 1][1]
                data_dict['ga_index'][k - 1].append([jj, ii, pos_assigned_gt_inds[i].item()])
                # print(index, ii, jj)
                box = pos_bboxes[i].numpy().astype(np.int32).tolist() + [pos_assigned_gt_inds[i].item()]
                if box not in data_dict['ga_bbox'][k - 1]:
                    data_dict['ga_bbox'][k - 1].append(box)

            for i in range(len(self.strides)):
                data_dict['nano_pnc'][i][1] = featmap_sizes[i][0] * featmap_sizes[i][1] - data_dict['nano_pnc'][i][0]
            # print(data_dict['ga_bbox'], len(data_dict['ga_bbox']))
            # print(data_dict['ga_index'], len(data_dict['ga_index']))
            # print(data_dict['nano_pnc'])
            # exit()

        # exit()
        return data_dict

    def __repr__(self):
        return 'CalcNanoGrids(scale={}, strides={}, top_k={}, num_classes={}, analysis={})'.format(self.scale, self.strides, self.top_k, self.num_classes, self.analysis)
