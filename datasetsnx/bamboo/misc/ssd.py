import torch
import numpy as np
from ..base_internode import BaseInternode
from utils.bbox_tools import xywh2xyxy, xyxy2xywh, ori_ssd_encode1, get_ssd_priors, calc_iou1


__all__ = ['CalcSSDGrids']


class CalcSSDGrids(BaseInternode):
    def __init__(self, num_classes, threshold=0.5, deta=0, loc=True):
        self.threshold = threshold

        self.deta = deta
        self.num_classes = num_classes
        self.label_neg = self.deta / (self.num_classes - 1)
        self.loc = loc

        self.image_size = kwargs['priors']['image_size']
        self.variances = kwargs['priors']['variances']
        self.priors = get_ssd_priors(**kwargs['priors'])
        self.priors_c = self.priors.copy()
        self.priors = xywh2xyxy(self.priors)

    def __call__(self, data_dict):
        gt = data_dict['bbox'].copy()

        gt[:, :4] /= self.image_size
        iou = calc_iou1(np.expand_dims(gt[:, :4], axis=0), np.expand_dims(self.priors, axis=1)).T
        best_prior_iou, best_prior_idx = np.max(iou, axis=1), np.argmax(iou, axis=1)
        best_gt_iou, best_gt_idx = np.max(iou, axis=0), np.argmax(iou, axis=0)

        best_gt_iou[best_prior_idx] = 2
        for j in range(best_prior_idx.shape[0]):
            best_gt_idx[best_prior_idx[j]] = j

        matches = gt[best_gt_idx]
        matches[:, 4][best_gt_iou < self.threshold] = 0

        if self.loc:
            matches[:, :4] = xyxy2xywh(matches[:, :4])
            matches[:, :4] = ori_ssd_encode1(matches[:, :4], self.priors_c, self.variances)
        else:
            matches[:, :4] *= self.image_size

        matches = torch.from_numpy(matches).type(torch.float32)
        loc = matches[:, :4]
        weight = matches[:, 5:6]

        label = matches[:, 4]
        p = label.clone().view(-1, 1).float()
        lp = torch.where(p < 1, p + 1, torch.tensor(1.0 - self.deta))
        smooth_label = torch.zeros(matches.shape[0], self.num_classes).scatter_(1, label.view(-1, 1).long(), lp)
        smooth_label[smooth_label[:, 0] == 0, 1:] += self.label_neg

        data_dict['gt'] = torch.cat([loc, smooth_label, weight], dim=1)

        return data_dict

    def __repr__(self):
        return 'CalcSSDGrids(threshold={}, num_classes={}, deta={},loc={})'.format(self.threshold, self.num_classes, self.deta, self.loc)

    def rper(self):
        return 'CalcSSDGrids(not available)'
