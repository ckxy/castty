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
from part_of_hitogata.config import load_config, load_config_far_away
from part_of_hitogata.utils.point_tools import draw_point
from part_of_hitogata.utils.heatmap_tools import draw_heatmap
from part_of_hitogata.utils.bbox_tools import draw_bbox

from torchvision.transforms.functional import to_tensor
from torch.nn.functional import interpolate


def hm1(data_dict, rc, index=0):
    heatmaps = data_dict['heatmap'][index]

    ori_size = data_dict['image_meta'][index]['ori_size']

    res = rc(image=data_dict['image'][index], ori_size=ori_size, heatmap=heatmaps)
    img = res['image']
    # p = res['point']
    heatmaps = res['heatmap']
    # heatmaps = heatmaps.unsqueeze(0)
    print(heatmaps.shape)
    # exit()
    # print(data_dict['point'][index])
    # print(data_dict['point'][index] - p)
    # print((data_dict['point'][index] - p).mean())
    # exit()

    # cols = math.ceil((1 + len(heatmaps)) / 4)
    # plt.subplot(3, 1, 1)
    # plt.imshow(img)
    # plt.axis('off') 

    # for i, heatmap in enumerate(heatmaps):
    #     tmp = draw_heatmap(heatmap)
    #     tmp = tmp.resize(img.size, Image.BILINEAR)

    #     res = Image.blend(img, tmp, 0.5)
    #     plt.subplot(3, 1, i + 2)
    #     plt.imshow(res)
    #     plt.axis('off') 

    for i, heatmap in enumerate(heatmaps):
        tmp = draw_heatmap(heatmap)
        # tmp.save('{}.jpg'.format(i))
        print(img.size, tmp.size)
        tmp = tmp.resize(img.size)

        res = Image.blend(img, tmp, 0.5)
        plt.subplot(4, 4, i + 1)
        plt.imshow(res)
        plt.axis('off')


def hm2(data_dict, rc, index=0):
    heatmaps = data_dict['center_heatmap'][index]
    points = data_dict['point'][index].numpy()

    print(data_dict['bbox'][index])
    print(data_dict['bbox_meta'][index])
    # exit()

    ori_size = data_dict['image_meta'][index]['ori_size']

    res = rc(image=data_dict['image'][index], ori_size=ori_size, point=points)
    img = res['image']
    points = res['point']

    img = draw_point(img, points, data_dict['point_meta'][index].get('keep', None))

    # heatmaps = res['heatmap']
    heatmaps = interpolate(heatmaps.unsqueeze(0), scale_factor=4, mode='bilinear', align_corners=False)[0]
    # print(heatmaps.shape)

    flag = torch.sum(heatmaps, dim=(-2, -1))
    flag = torch.nonzero(flag)[..., 0].numpy().tolist()

    for i, j in enumerate(flag):
        tmp = draw_heatmap(heatmaps[j])
        tmp = rc(image=to_tensor(tmp), ori_size=ori_size)['image']

        print(img.size, tmp.size)
        tmp = tmp.resize(img.size)

        res = Image.blend(img, tmp, 0.5)

        plt.subplot(1, 1, i + 1)
        plt.imshow(res)
        plt.axis('off')


def hm3(data_dict, classes, rc, index=0):
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

    # plt.subplot(1, 3, 1)
    plt.imshow(pt_img)
    plt.axis('off')

    # center_heatmaps = data_dict['center_heatmap'][index]
    # center_heatmaps = interpolate(center_heatmaps.unsqueeze(0), scale_factor=4, mode='bilinear', align_corners=False)[0]

    # tmp = draw_heatmap(center_heatmaps[0])
    # tmp = rc(image=to_tensor(tmp), ori_size=data_dict['ori_size'][index])['image']

    # cen_img = Image.blend(img, tmp, 0.5)

    # plt.subplot(1, 3, 2)
    # plt.imshow(cen_img)
    # plt.axis('off')

    # kpt_heatmaps = data_dict['kpt_heatmap'][index]
    # kpt_heatmaps = interpolate(kpt_heatmaps.unsqueeze(0), scale_factor=4, mode='bilinear', align_corners=False)[0]

    # tmp = draw_heatmap(kpt_heatmaps[0])
    # tmp = rc(image=to_tensor(tmp), ori_size=data_dict['ori_size'][index])['image']

    # kpt_img = Image.blend(img, tmp, 0.5)

    # plt.subplot(1, 3, 3)
    # plt.imshow(kpt_img)
    # plt.axis('off')


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    # torch.set_printoptions(precision=4, threshold=None, edgeitems=None, linewidth=None, profile=None)

    cfg = load_config_far_away('configs/heatmap.py')
    data_manager = DataManager(cfg.test_data)
    dataloader = data_manager.load_data()
    info = data_manager.info

    time.sleep(0.1)

    rc = data_manager.oobmab
    print(data_manager)
    print(data_manager.dataset.bamboo.rper())
    # exit()

    for data in tqdm(dataloader):
        hm2(data, rc, 0)
        plt.show()
        break

