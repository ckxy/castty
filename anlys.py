import os
import cv2
import math
import time
import torch
import ntpath
from PIL import Image
import numpy as np
from configs import load_config, load_config_far_away
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.bbox_tools import xywh2xyxy, xyxy2xywh, draw_bbox, grid_analysis

from datasetsnx import create_data_manager
from visualization.visualizer import Visualizer


def merge_dict(src, dst):
    for k in src.keys():
        if k not in dst.keys():
            dst[k] = src[k]
        else:
            dst[k].update(src[k])
    return dst


def calc_img_wh(img_w, img_h, heat=True, marginal=True):
    img_wh = np.concatenate((np.array(img_w)[..., np.newaxis], np.array(img_h)[..., np.newaxis]), axis=-1)

    vis_dict = dict(
        scatter=dict(
            oridatainfo_imgwh=dict(
                x=img_wh,
                y=np.ones(len(img_wh)).astype(np.int),
                opts=dict(title='oridatainfo_imgwh', markersize=1, webgl=True),
            ),
        ),
    )

    if heat:
        max_w = int(np.ceil(np.max(img_wh[..., 0]) / 100) * 100)
        max_h = int(np.ceil(np.max(img_wh[..., 1]) / 100) * 100)
        heat = np.zeros((max_w // 100 + 1, max_h // 100 + 1))
        tmp = np.round(img_wh / 100).astype(np.int)
        for t in tmp:
            heat[t[0], t[1]] += 1

        vis_dict['heatmap'] = dict()

        vis_dict['heatmap']['oridatainfo_imgwh_heat'] = dict(
            x=heat,
            opts=dict(
                title='oridatainfo_imgwh_heat', 
                xtickvals=[i for i in range(max_w)],
                xticklabels=[str(i * 100) for i in range(max_w)],
                ytickvals=[i for i in range(max_h)],
                yticklabels=[str(i * 100) for i in range(max_h)],
            ),
        )

    if marginal:
        vis_dict['histogram'] = dict(
            oridatainfo_imgw=dict(
                x=np.array(img_w),
                opts=dict(title='oridatainfo_imgw', numbins=20, webgl=True),
            ),
            oridatainfo_imgh=dict(
                x=np.array(img_h),
                opts=dict(title='oridatainfo_imgh', numbins=20, webgl=True),
            ),
        )
    return vis_dict


def calc_box_wh(box_w, box_h, heat=True, marginal=True):
    box_w_tmp = []
    for w in box_w:
        box_w_tmp += w.tolist()
    box_h_tmp = []
    for h in box_h:
        box_h_tmp += h.tolist()
    box_wh = np.concatenate((np.array(box_w_tmp)[..., np.newaxis], np.array(box_h_tmp)[..., np.newaxis]), axis=-1)

    vis_dict = dict(
        scatter=dict(
            oridatainfo_boxwh=dict(
                x=box_wh,
                y=np.ones(len(box_wh)).astype(np.int),
                opts=dict(title='oridatainfo_boxwh', markersize=1, webgl=True),
            ),
        ),
    )

    if heat:
        max_w = int(np.ceil(np.max(box_wh[..., 0]) / 10) * 10)
        max_h = int(np.ceil(np.max(box_wh[..., 1]) / 10) * 10)

        max_v = max(max_w, max_h)
        s = max(10, math.ceil(max_v / 110) * 10)

        heat = np.zeros((max_w // 10 + 1, max_h // 10 + 1))
        tmp = np.round(box_wh / 10).astype(np.int)
        for t in tmp:
            heat[t[0], t[1]] += 1

        vis_dict['heatmap'] = dict()

        vis_dict['heatmap']['oridatainfo_boxwh_heat'] = dict(
            x=heat,
            opts=dict(
                title='oridatainfo_boxwh_heat',
                ylabel='x10',
                xlabel='x10',
            ),
        )

    if marginal:
        vis_dict['histogram'] = dict(
            oridatainfo_boxw=dict(
                x=box_wh[..., 0],
                opts=dict(title='oridatainfo_boxw', numbins=100, webgl=True),
            ),
            oridatainfo_boxh=dict(
                x=box_wh[..., 1],
                opts=dict(title='oridatainfo_boxh', numbins=100, webgl=True),
            ),
        )
    return vis_dict


def calc_box_whir(box_w, box_h, img_w, img_h, marginal=True):
    box_w_tmp = []
    for bw, iw in zip(box_w, img_w):
        box_w_tmp += (bw / iw).tolist()
    box_h_tmp = []
    for bh, ih in zip(box_h, img_h):
        box_h_tmp += (bh / ih).tolist()
    box_whir = np.concatenate((np.array(box_w_tmp)[..., np.newaxis], np.array(box_h_tmp)[..., np.newaxis]), axis=-1)

    n = 10
    heat = np.zeros((n + 1, n + 1))
    tmp = np.round(box_whir * n).astype(np.int)
    for t in tmp:
        heat[t[0], t[1]] += 1

    vis_dict = dict(
        heatmap=dict(
            oridatainfo_boxwhir=dict(
                x=heat,
                opts=dict(
                    title='oridatainfo_boxwhir', 
                    xtickvals=[i for i in range(n + 1)],
                    xticklabels=[str(i / n) for i in range(n + 1)],
                    ytickvals=[i for i in range(n + 1)],
                    yticklabels=[str(i / n) for i in range(n + 1)],
                ),
            ),
        ),
    )

    if marginal:
        vis_dict['histogram'] = dict(
            oridatainfo_boxwir=dict(
                x=np.array(box_w_tmp),
                opts=dict(title='oridatainfo_boxwir', numbins=100, webgl=True),
            ),
            oridatainfo_boxhir=dict(
                x=np.array(box_h_tmp),
                opts=dict(title='oridatainfo_boxhir', numbins=100, webgl=True),
            ),
            oridatainfo_boxwhr=dict(
                x=np.array(box_w_tmp) / np.array(box_h_tmp),
                opts=dict(title='oridatainfo_boxwhr', numbins=100, webgl=True),
            ),
        )
    return vis_dict


def calc_box_cir(box_cx, box_cy, img_w, img_h):
    box_cx_tmp = []
    for x, iw in zip(box_cx, img_w):
        box_cx_tmp += (x / iw).tolist()
    box_cy_tmp = []
    for y, ih in zip(box_cy, img_h):
        box_cy_tmp += (y / ih).tolist()
    box_c = np.concatenate((np.array(box_cx_tmp)[..., np.newaxis], np.array(box_cy_tmp)[..., np.newaxis]), axis=-1)

    n = 10
    heat = np.zeros((n + 1, n + 1))
    tmp = np.round(box_c * n).astype(np.int)
    for t in tmp:
        heat[t[0], t[1]] += 1

    vis_dict = dict(
        heatmap=dict(
            oridatainfo_boxcir=dict(
                x=heat,
                opts=dict(
                    title='oridatainfo_boxcir', 
                    xtickvals=[i for i in range(n + 1)],
                    xticklabels=[str(i / n) for i in range(n + 1)],
                    ytickvals=[i for i in range(n + 1)],
                    yticklabels=[str(i / n) for i in range(n + 1)],
                ),
            ),
        ),
    )
    return vis_dict


def calc_box_clscount(box_cls, num_classes):
    box_cls_count = np.zeros(num_classes).astype(np.int)
    for cs in box_cls:
        for c in cs.tolist():
            box_cls_count[c] += 1

    vis_dict = dict(
        bar=dict(
            oridatainfo_cls=dict(
                x=box_cls_count,
                opts=dict(title='oridatainfo_cls'),
            ),
        ),
    )
    return vis_dict


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    # torch.set_printoptions(precision=4, threshold=None, edgeitems=None, linewidth=None, profile=None)

    cfg = load_config('spnx')
    vis = Visualizer(cfg)

    # vis.vis.bar(X=np.random.rand(20))
    # exit()

    # data_manager = create_data_manager(cfg.train_data)
    data_manager = create_data_manager(cfg.test_data)
    dataloader = data_manager.load_data()
    info = data_manager.info
    
    time.sleep(0.1)

    rc = info['oobmab']
    print(data_manager.dataset.bamboo.rper())
    # print(data_manager.dataset.get_data_info(0))

    classes = info['classes']

    box_cls = []
    box_w = []
    box_h = []
    box_cx = []
    box_cy = []
    img_w = []
    img_h = []
    for i in tqdm(range(len(data_manager.dataset))):
        di = data_manager.dataset.get_data_info(i)
        boxes = di['bbox']
        boxes = xyxy2xywh(boxes)

        img_w.append(di['w'])
        img_h.append(di['h'])

        box_cx.append(boxes[:, 0])
        box_cy.append(boxes[:, 1])
        box_w.append(boxes[:, 2])
        box_h.append(boxes[:, 3])
        box_cls.append(boxes[:, 4].astype(np.int))

    vis_dict = dict()

    vis_dict = merge_dict(calc_img_wh(img_w, img_h), vis_dict)
    vis_dict = merge_dict(calc_box_wh(box_w, box_h), vis_dict)
    vis_dict = merge_dict(calc_box_whir(box_w, box_h, img_w, img_h), vis_dict)
    vis_dict = merge_dict(calc_box_cir(box_cx, box_cy, img_w, img_h), vis_dict)
    vis_dict = merge_dict(calc_box_clscount(box_cls, len(info['classes'])), vis_dict)

    # print(vis_dict.keys())
    # exit()

    vis.visualize(vis_dict, 0, 0)
    # plt.scatter(img_w, img_h, s=2, alpha=0.6, color='red')
    # plt.show()
