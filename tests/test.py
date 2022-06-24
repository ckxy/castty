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
from part_of_hitogata.utils.bbox_tools import xywh2xyxy, xyxy2xywh, draw_bbox, grid_analysis
from part_of_hitogata.utils.polygon_tools import draw_polygon_without_label


def ga(data_dict, classes, rc, index=0):
    bboxes = data_dict['bbox'][index].cpu().numpy()
    print(data_dict['image'][index].shape)

    print(bboxes, 'b')

    res = rc(image=data_dict['image'][index], ori_size=data_dict['ori_size'][index], bbox=bboxes)
    img1 = res['image']
    img2 = img1.copy()
    bboxes = res['bbox']

    if 'ga_bbox' in data_dict:
        ga_img = grid_analysis(img1, (8, 16, 32), data_dict['ga_bbox'][index], data_dict['ga_index'][index], len(bboxes))

    b_img = draw_bbox(img2, res['bbox'], data_dict['bbox_meta'][index].get('score'), data_dict['bbox_meta'][index].get('class_id'), classes)
    # b_img.save('1.jpg')
    # ga_img.save('atssa.jpg')

    print(bboxes, 'a')
    print(b_img, data_dict['ori_size'][index])

    if 'ga_bbox' in data_dict:
        plt.subplot(211)
        plt.imshow(b_img)
        plt.axis('off')
        plt.subplot(212)
        plt.imshow(ga_img)
        plt.axis('off')
    else:
        plt.imshow(b_img)
        plt.axis('off')


def po(data_dict, rc, index=0):
    poly = data_dict['poly'][index]
    print(data_dict['image'][index].shape, data_dict['path'][index])

    # print(data_dict['poly'], '000')
    print(poly, 'b')
    print(data_dict['poly_meta'][index])

    a = np.array([[260, 497], [365, 497], [365, 588], [260, 588]]).astype(np.float32)
    poly[-1] = a

    res = rc(image=data_dict['image'][index], ori_size=data_dict['ori_size'][index], poly=poly)
    img = res['image']
    poly = res['poly']

    print(data_dict['ori_size'][index])
    print(poly, 'a')

    img = draw_polygon_without_label(img, poly, data_dict['poly_meta'][index])

    plt.figure()
    # plt.subplot(121)
    plt.imshow(img)
    plt.axis('off')

    # plt.subplot(122)
    # plt.imshow(data_dict['ocrdet_train_mask'][0].numpy(), cmap='gray')
    # plt.axis('off')

    # plt.figure()
    # for i in range(6):
    #     plt.subplot(2, 3, i + 1)
    #     plt.imshow(data_dict['ocrdet_kernel'][0][i].numpy(), cmap='gray')
    #     plt.axis('off')


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    # torch.set_printoptions(precision=4, threshold=None, edgeitems=None, linewidth=None, profile=None)

    # cfg = load_config('spnx')
    cfg = load_config_far_away('configs/spnx.py')
    data_manager = DataManager(cfg.test_data)
    dataloader = data_manager.load_data()
    info = data_manager.info

    time.sleep(0.1)

    rc = info['oobmab']
    print(data_manager)
    print(data_manager.dataset.bamboo.rper())
    # exit()

    for data in tqdm(dataloader):
        ga(data, dataloader.dataset.info.classes, rc, 0)
        # po(data, rc, 0)
        plt.show()
        break

