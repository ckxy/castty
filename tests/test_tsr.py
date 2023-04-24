import os
import cv2
import math
import time
import torch
import ntpath
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from part_of_hitogata.datasets import DataManager
from part_of_hitogata.configs import load_config, load_config_far_away
from part_of_hitogata.utils.point_tools import draw_point, draw_point_without_label
from part_of_hitogata.utils.heatmap_tools import draw_heatmap
from part_of_hitogata.utils.bbox_tools import draw_bbox, draw_bbox_without_label
from part_of_hitogata.utils.mask_tools import draw_mask

from torchvision.transforms.functional import to_tensor
from torch.nn.functional import interpolate
from PIL import Image, ImageDraw, ImageFont


def draw_tsr(img, l_row, c_row, l_col, c_col):
    if not isinstance(img, Image.Image):
        is_np = True
        img = Image.fromarray(img)
    else:
        is_np = False

    w, h = img.size
    l = math.sqrt(h * h + w * w)
    r = max(2, int(l / 200))

    row_vis = img.copy()
    draw = ImageDraw.Draw(row_vis)
    rsplit = l_row.shape[1] + 1
    gt_l_row = (l_row.cpu().numpy() * h).astype(np.int32)

    xs = np.linspace(0, w, num=rsplit + 1)[1:-1]

    for i in range(l_row.shape[0]):
        for j in range(l_row.shape[1] - 1):
            x1, x2 = int(xs[j]), int(xs[j + 1])

            y1, y2 = gt_l_row[i, j, 0], gt_l_row[i, j + 1, 0]
            draw.line((x1, y1, x2, y2), width=r // 2, fill=(255, 100, 120))

            y1, y2 = gt_l_row[i, j, 1], gt_l_row[i, j + 1, 1]
            draw.line((x1, y1, x2, y2), width=r // 2, fill=(255, 0, 0))

            y1, y2 = gt_l_row[i, j, 2], gt_l_row[i, j + 1, 2]
            draw.line((x1, y1, x2, y2), width=r // 2, fill=(255, 100, 100))

        for j in range(l_row.shape[1]):
            x = int(xs[j])
            y1, y2, y3 = gt_l_row[i, j]
            if c_row[i, j]:
                draw.ellipse((x - r, y1 - r, x + r, y1 + r), fill=(255, 100, 100))
                draw.ellipse((x - r, y2 - r, x + r, y2 + r), fill=(255, 0, 0))
                draw.ellipse((x - r, y3 - r, x + r, y3 + r), fill=(255, 100, 100))
            else:
                draw.ellipse((x - r, y1 - r, x + r, y1 + r), fill=(100, 100, 255))
                draw.ellipse((x - r, y2 - r, x + r, y2 + r), fill=(0, 0, 255))
                draw.ellipse((x - r, y3 - r, x + r, y3 + r), fill=(100, 100, 255))

    col_vis = img.copy()
    draw = ImageDraw.Draw(col_vis)
    csplit = l_col.shape[1] + 1
    gt_l_col = (l_col.cpu().numpy() * w).astype(np.int32)

    ys = np.linspace(0, h, num=csplit + 1)[1:-1]

    for i in range(l_col.shape[0]):
        for j in range(l_col.shape[1] - 1):
            y1, y2 = int(ys[j]), int(ys[j + 1])

            x1, x2 = gt_l_col[i, j, 0], gt_l_col[i, j + 1, 0]
            draw.line((x1, y1, x2, y2), width=r // 2, fill=(255, 100, 120))

            x1, x2 = gt_l_col[i, j, 1], gt_l_col[i, j + 1, 1]
            draw.line((x1, y1, x2, y2), width=r // 2, fill=(255, 0, 0))

            x1, x2 = gt_l_col[i, j, 2], gt_l_col[i, j + 1, 2]
            draw.line((x1, y1, x2, y2), width=r // 2, fill=(255, 100, 100))

        for j in range(l_col.shape[1]):
            y = int(ys[j])
            x1, x2, x3 = gt_l_col[i, j]
            if c_col[i, j]:
                draw.ellipse((x1 - r, y - r, x1 + r, y + r), fill=(255, 100, 100))
                draw.ellipse((x2 - r, y - r, x2 + r, y + r), fill=(255, 0, 0))
                draw.ellipse((x3 - r, y - r, x3 + r, y + r), fill=(255, 100, 100))
            else:
                draw.ellipse((x1 - r, y - r, x1 + r, y + r), fill=(100, 100, 255))
                draw.ellipse((x2 - r, y - r, x2 + r, y + r), fill=(0, 0, 255))
                draw.ellipse((x3 - r, y - r, x3 + r, y + r), fill=(100, 100, 255))

    if is_np:
        row_vis = np.array(row_vis)
        col_vis = np.array(col_vis)

    return row_vis, col_vis


def tsr(data_dict, classes, rc, index=0):
    _, h, w = data_dict['image'][index].shape
    points = data_dict['point'][index]
    bboxes = data_dict['bbox'][index]
    row_mask = data_dict['tsr_row_mask'][index]
    col_mask = data_dict['tsr_col_mask'][index]

    l_row = data_dict['tsr_l_row'][index]
    c_row = data_dict['tsr_c_row'][index]

    l_col = data_dict['tsr_l_col'][index]
    c_col = data_dict['tsr_c_col'][index]

    # print(bboxes, 'bbb')
    # print(points, 'ppp')
    # print(data_dict['bbox_meta'][index])

    res = rc(image=data_dict['image'][index], ori_size=data_dict['ori_size'][index], point=points, bbox=bboxes, mask=row_mask)
    img = res['image']
    points = res['point']
    bboxes = res['bbox']
    row_mask = res['mask']

    res = rc(ori_size=data_dict['ori_size'][index], mask=col_mask)
    col_mask = res['mask']

    pt_img = draw_point_without_label(img.copy(), points, data_dict['point_meta'][index].get('visible', None))
    row_mask, _ = draw_mask(img.copy(), row_mask, ['1'], colorbar=False)
    col_mask, _ = draw_mask(img.copy(), col_mask, ['1'], colorbar=False)

    # print(l_row * h, l_row.shape)
    row_vis, col_vis = draw_tsr(img.copy(), l_row, c_row, l_col, c_col)

    plt.subplot(2, 3, 1)
    plt.imshow(pt_img)
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(row_mask)
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(col_mask)
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(row_vis)
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(col_vis)
    plt.axis('off')

    # plt.subplot(2, 2, 1)
    # plt.imshow(pt_img)
    # plt.axis('off')

    # plt.subplot(2, 2, 2)
    # plt.imshow(row_mask)
    # plt.axis('off')

    # plt.subplot(2, 2, 4)
    # plt.imshow(row_vis)
    # plt.axis('off')


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    # torch.set_printoptions(precision=4, threshold=None, edgeitems=None, linewidth=None, profile=None)

    cfg = load_config_far_away('configs/point.py')
    data_manager = DataManager(cfg.test_data)
    dataloader = data_manager.load_data()
    info = data_manager.info

    time.sleep(0.1)

    rc = data_manager.oobmab
    print(data_manager)
    print(data_manager.dataset.bamboo.rper())
    # exit()

    for data in tqdm(dataloader):
        # pass
        tsr(data, info['bbox_classes'], rc, 0)
        plt.show()
        break

