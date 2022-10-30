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


# def kp(data_dict, rc, index=0):
#     print(data_dict['image'][index].shape)
#     bboxes = data_dict['bbox'][index].cpu().numpy()
#     points = data_dict['point'][index].numpy()

#     res = rc(image=data_dict['image'][index], ori_size=data_dict['ori_size'][index], point=points, bbox=bboxes)
#     img = res['image']
#     points = res['point']
#     bboxes = res['bbox']

#     print(os.path.splitext(ntpath.basename(data_dict['path'][index]))[0])
#     print(data_dict['path'][index])
#     # print(data_dict['euler_angle'][index])

#     img = draw_point(img, points, data_dict['point_meta'][index]['visible'])
#     img = draw_bbox(img, bboxes)
#     plt.imshow(img)
#     plt.axis('off')


def kp(data_dict, rc, index=0):
    print(data_dict['image'][index].shape)
    points = data_dict['point'][index].numpy()

    res = rc(image=data_dict['image'][index], ori_size=data_dict['ori_size'][index], point=points)
    img = res['image']
    points = res['point']

    print(os.path.splitext(ntpath.basename(data_dict['path'][index]))[0])
    print(data_dict['path'][index])
    # print(data_dict['euler_angle'][index])

    img = draw_point(img, points, data_dict['point_meta'][index].get('visible', None))
    plt.imshow(img)
    plt.axis('off')


def hm(data_dict, rc, index=0):
    heatmaps = data_dict['heatmap'][index]

    res = rc(image=data_dict['image'][index], ori_size=data_dict['ori_size'][index], heatmap=heatmaps)
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

        res = Image.blend(img, tmp, 0.5)
        plt.subplot(4, 4, i + 1)
        plt.imshow(res)
        plt.axis('off')


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
        kp(data, rc, 0)
        # hm(data, rc, 0)
        plt.show()
        break

