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
from part_of_hitogata.utils.point_tools import draw_point
from part_of_hitogata.utils.heatmap_tools import draw_heatmap
from part_of_hitogata.utils.bbox_tools import draw_bbox

from torchvision.transforms.functional import to_tensor
from torch.nn.functional import interpolate


def hm(data_dict, rc, index=0):
    heatmaps = data_dict['center_heatmap'][index]
    points = data_dict['point'][index]

    print(data_dict['bbox'][index])
    print(data_dict['bbox_meta'][index])
    # exit()

    res = rc(image=data_dict['image'][index], ori_size=data_dict['ori_size'][index], point=points)
    img = res['image']
    points = res['point']

    img = draw_point(img, points, data_dict['point_meta'][index].get('visible', None))

    # heatmaps = res['heatmap']
    heatmaps = interpolate(heatmaps.unsqueeze(0), scale_factor=4, mode='bilinear', align_corners=False)[0]
    # print(heatmaps.shape)

    flag = torch.sum(heatmaps, dim=(-2, -1))
    flag = torch.nonzero(flag)[..., 0].numpy().tolist()

    for i, j in enumerate(flag):
        tmp = draw_heatmap(heatmaps[j])
        tmp = rc(image=to_tensor(tmp), ori_size=data_dict['ori_size'][index])['image']

        res = Image.blend(img, tmp, 0.5)

        plt.subplot(1, 1, i + 1)
        plt.imshow(res)
        plt.axis('off')


def hm2(data_dict, classes, rc, index=0):
    points = data_dict['point'][index]
    bboxes = data_dict['bbox'][index]

    print(bboxes, 'bbb')
    print(data_dict['bbox_meta'][index])

    res = rc(image=data_dict['image'][index], ori_size=data_dict['ori_size'][index], point=points, bbox=bboxes)
    img = res['image']
    points = res['point']
    bboxes = res['bbox']

    pt_img = draw_point(img.copy(), points, data_dict['point_meta'][index].get('visible', None))
    pt_img = draw_bbox(pt_img, bboxes, data_dict['bbox_meta'][index].get('class_id', None), classes, None)

    # plt.subplot(1, 2, 1)
    plt.imshow(pt_img)
    plt.axis('off')

    # center_heatmaps = data_dict['center_heatmap'][index]
    # center_heatmaps = interpolate(center_heatmaps.unsqueeze(0), scale_factor=4, mode='bilinear', align_corners=False)[0]

    # tmp = draw_heatmap(center_heatmaps[0])
    # tmp = rc(image=to_tensor(tmp), ori_size=data_dict['ori_size'][index])['image']

    # cen_img = Image.blend(img, tmp, 0.5)

    # plt.subplot(1, 2, 2)
    # plt.imshow(cen_img)
    # plt.axis('off')

    # kpt_heatmaps = data_dict['kpt_heatmap'][index]
    # kpt_heatmaps = interpolate(kpt_heatmaps.unsqueeze(0), scale_factor=4, mode='bilinear', align_corners=False)[0]

    # for i in range(16):
    #     tmp = draw_heatmap(kpt_heatmaps[i])
    #     tmp = rc(image=to_tensor(tmp), ori_size=data_dict['ori_size'][index])['image']

    #     kpt_img = Image.blend(img, tmp, 0.5)

    #     plt.subplot(4, 4, i + 1)
    #     plt.imshow(kpt_img)
    #     plt.axis('off')


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
        # kp(data, rc, 0)
        hm2(data, info['bbox_classes'], rc, 0)
        plt.show()
        break

