import math
import torch
import numpy as np
from ..builder import INTERNODE
from torchvision.ops.boxes import box_area
from ..base_internode import BaseInternode
from ...utils.common import get_image_size
from ....utils.bbox_tools import xyxy2xywh, xywh2xyxy, calc_iou2


__all__ = ['CalcDSLabel']


def reorder(cla, box, o="xyxy", max_elem=None):
    if o == "cxcywh":
        box = xywh2xyxy(box)
    if max_elem == None:
        max_elem = len(cla)
    # init
    order = []
    
    # convert
    cla = np.array(cla)
    area = box_area(box)
    order_area = sorted(list(enumerate(area)), key=lambda x: x[1], reverse=True)
    iou = calc_iou2(box.unsqueeze(0), box.unsqueeze(1))
    
    # arrange
    text = np.where(cla == 0)[0]
    logo = np.where(cla == 1)[0]
    deco = np.where(cla == 2)[0]
    order_text = sorted(np.array(list(enumerate(area)))[text].tolist(), key=lambda x: x[1], reverse=True)
    order_deco = sorted(np.array(list(enumerate(area)))[deco].tolist(), key=lambda x: x[1])
    
    # deco connection
    connection = {}
    reverse_connection = {}
    for idx, _ in order_deco:
        idx = int(idx)
        con = []
        for idx_ in logo:
            if iou[idx, idx_]:
                connection[idx_] = idx
                con.append(idx_)
        for idx_ in text:
            if iou[idx, idx_]:
                connection[idx_] = idx
                con.append(idx_)
        for idx_ in deco:
            if idx == idx_: continue
            if iou[idx, idx_]:
                if idx_ not in connection:
                    connection[idx_] = [idx]
                else:
                    connection[idx_].append(idx)
                con.append(idx_)
        reverse_connection[idx] = con
                    
    # reorder
    for idx in logo:
        if idx in connection:
            d = connection[idx]
            d_group = reverse_connection[d]
            for idx_ in d_group:
                if idx_ not in order:
                    order.append(idx_)
            if d not in order:
                order.append(d)
        else:
            order.append(idx)
    for idx, _ in order_text:
        if len(order) >= max_elem:
            break
        if idx in connection:
            d = connection[idx]
            d_group = reverse_connection[d]
            for idx_ in d_group:
                if idx_ not in order:
                    order.append(idx_)
            if d not in order:
                order.append(d)
        else:
            order.append(idx)
            
    if len(order) < max_elem:
        non_obj = np.where(cla == 0)[0]
        order.extend(non_obj)

    order = order[:min(len(cla), max_elem)]
    order = [int(i) for i in order]
    return order


@INTERNODE.register_module()
class CalcDSLabel(BaseInternode):
    def __init__(self, max_elem=32, **kwargs):
        self.max_elem = max_elem

        BaseInternode.__init__(self, **kwargs)

    def forward(self, data_dict, **kwargs):
        w, h = get_image_size(data_dict['image'])
        bboxes = torch.from_numpy(data_dict['bbox']).type(torch.float32)
        cla = data_dict['bbox_meta']['class_id']

        order = reorder(cla, bboxes, "xyxy", self.max_elem)

        label = np.zeros((self.max_elem, 2, 4), dtype=np.float32)

        for i in range(len(order)):
            idx = order[i]
            label[i][0][int(cla[idx])] = 1
            label[i][1] = bboxes[idx]
            if label[i][1][0] > label[i][1][2] or label[i][1][1] > label[i][1][3]:
                label[i][1][:2], label[i][1][2:] = label[i][1][2:], label[i][1][:2]
            label[i][1] = xyxy2xywh(torch.tensor(label[i][1]))
            label[i][1][::2] /= w
            label[i][1][1::2] /= h
        for i in range(len(order), self.max_elem):
            label[i][0][0] = 1

        label = torch.tensor(label).type(torch.float32)
        data_dict['ds_label'] = label

        return data_dict

    def __repr__(self):
        return f'CalcDSLabel(max_elem={self.max_elem})'


@INTERNODE.register_module()
class DSMerge(BaseInternode):
    def forward(self, data_dict, **kwargs):
        data_dict['ds_image'] = torch.concat([data_dict['image'], data_dict['saliency_map'][:1, ...]])
        return data_dict

    def __repr__(self):
        return 'DSMerge()'
