import numpy as np
from ..base_internode import BaseInternode
from utils.bbox_tools import get_xy_grid1


__all__ = ['CalcGrids']


def calc_centerness(offsets):
    left_right = offsets[:, [0, 2]]
    top_bottom = offsets[:, [1, 3]]
    centerness = (left_right.min(axis=-1) / left_right.max(axis=-1)) * \
                  (top_bottom.min(axis=-1) / top_bottom.max(axis=-1))
    return np.sqrt(centerness)


class CalcGrids(BaseInternode):
    def __init__(self, strides, num_classes, sml_thresh, gt_per_grid=3, deta=0, centerness=0, analysis=False):
        assert len(strides) == 3
        assert len(sml_thresh) == 2

        self.strides = np.array(strides)
        self.gt_per_grid = gt_per_grid
        self.num_classes = num_classes
        self.deta = deta
        self.sml_thresh = sml_thresh
        self.analysis = analysis
        self.centerness = centerness

    def __call__(self, data_dict):
        output_h = data_dict['image'].shape[1] // self.strides
        output_w = data_dict['image'].shape[2] // self.strides

        temp_batch_sbboxes = []
        temp_batch_mbboxes = []
        temp_batch_lbboxes = []

        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes, ga_bbox, ga_index = \
            self.preprocess_anchorfree(data_dict, output_h, output_w)

        if self.analysis:
            data_dict['ga_bbox'] = ga_bbox
            data_dict['ga_index'] = ga_index

        data_dict['sbbox'] = np.array(sbboxes) if len(sbboxes) > 0 else np.zeros((0, 4))
        data_dict['mbbox'] = np.array(mbboxes) if len(mbboxes) > 0 else np.zeros((0, 4))
        data_dict['lbbox'] = np.array(lbboxes) if len(lbboxes) > 0 else np.zeros((0, 4))

        data_dict['label_sbbox'] = label_sbbox
        data_dict['label_mbbox'] = label_mbbox
        data_dict['label_lbbox'] = label_lbbox

        return data_dict

    def preprocess_anchorfree(self, data_dict, output_h, output_w):
        label = [np.zeros((output_h[i], output_w[i],
                           self.gt_per_grid, 6 + self.num_classes)).astype(np.float32) for i in range(3)]
        # mixup weight位默认为1.0
        for i in range(3):
            label[i][:, :, :, -1] = 1.0
        bboxes_coor = [[] for _ in range(3)]
        bboxes_count = [np.zeros((output_h[i], output_w[i])) for i in range(3)]
            
        ga_bbox = [[] for _ in range(3)]
        ga_index = [[] for _ in range(3)]

        grids = [(get_xy_grid1(output_h[i], output_w[i], 1).squeeze() + 0.5) * self.strides[i] for i in range(3)]

        bw = data_dict['bbox'][:, 3] - data_dict['bbox'][:, 1]
        bh = data_dict['bbox'][:, 2] - data_dict['bbox'][:, 0]
        bbox_scales = np.sqrt(bw * bh)
        max_ids = np.argsort(bbox_scales)[::-1]

        for i in max_ids:
            bbox = data_dict['bbox'][i]
            bbox_coor = bbox[:4]
            bbox_class_ind = int(bbox[4])
            bbox_cxy = (bbox_coor[2:] + bbox_coor[:2]) * 0.5
            bbox_scale = bbox_scales[i]
            bbox_mixw = bbox[5]
            
            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0

            # label smooth
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            smooth_onehot = onehot * (1 - self.deta) + self.deta * uniform_distribution
            # smooth_onehot = onehot
            
            if bbox_scale <= self.sml_thresh[0]:
                match_branch = 0
            elif self.sml_thresh[0] < bbox_scale <= self.sml_thresh[1]:
                match_branch = 1
            else:
                match_branch = 2

            if 0 < self.centerness <= 1:
                cx, cy = grids[match_branch][..., 0], grids[match_branch][..., 1]
                l = cx - bbox_coor[0]
                t = cy - bbox_coor[1]
                r = bbox_coor[2] - cx
                b = bbox_coor[3] - cy
                offsets = np.concatenate([l[..., np.newaxis], t[..., np.newaxis], r[..., np.newaxis], b[..., np.newaxis]], axis=-1)
                ind =  (l >= 0) * (t >= 0) * (r >= 0) * (b >= 0)
                nzy, nzx = np.nonzero(ind)
                s_offsets = offsets[nzy, nzx]
                s_centerness = calc_centerness(s_offsets)

                nz_scen_ind = np.nonzero(s_centerness >= (1 - self.centerness))[0].tolist()

                if len(nz_scen_ind) == 0:
                    nz_scen_ind.append(np.argmax(s_centerness))

                xids, yids = nzx[nz_scen_ind], nzy[nz_scen_ind]
            else:
                xind, yind = np.floor(1.0 * bbox_cxy / self.strides[match_branch]).astype(np.int32)
                xids, yids = [xind], [yind]

            # print('new', match_branch, xids, yids)

            for xind, yind in zip(xids, yids):
                gt_count = int(bboxes_count[match_branch][yind, xind])
                if gt_count < self.gt_per_grid:
                    if gt_count == 0:
                        gt_count = slice(None)
                    bbox_label = np.concatenate([bbox_coor, [1.0], smooth_onehot, [bbox_mixw]], axis=-1)
                    label[match_branch][yind, xind, gt_count, :] = bbox_label
                    bboxes_count[match_branch][yind, xind] += 1
                    bboxes_coor[match_branch].append(bbox_coor)

                    if self.analysis:
                        ga_bbox[match_branch].append(bbox_coor.astype(np.int).tolist() + [i])
                        ga_index[match_branch].append([xind, yind, i])

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_coor

        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes, ga_bbox, ga_index

    def __repr__(self):
            return 'CalcGrids(strides={}, gt_per_grid={}, num_classes={}, sml_thresh={}, deta={}, centerness={}, analysis={})'.format(tuple(self.strides), self.gt_per_grid, self.num_classes, self.sml_thresh, self.deta, self.centerness, self.analysis)

    def rper(self):
        return 'CalcGrids(not available)'
