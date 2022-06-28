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
from part_of_hitogata.utils.polygon_tools import *



def po(data_dict, rc, index=0):
    poly = data_dict['poly'][index]
    print(data_dict['image'][index].shape, data_dict['path'][index])

    # print(data_dict['poly'], '000')
    print(poly, 'b')
    print(data_dict['poly_meta'][index])

    print(data_dict['ocrdet_kernel'].shape)

    # a = np.array([[260, 497], [365, 497], [365, 588], [260, 588]]).astype(np.float32)
    # poly[-1] = a

    res = rc(image=data_dict['image'][index], ori_size=data_dict['ori_size'][index], poly=poly)
    img = res['image']
    poly = res['poly']

    print(data_dict['ori_size'][index])
    print(poly, 'a')

    img = draw_polygon_without_label(img, poly, data_dict['poly_meta'][index].get('ignore_flag'))

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


def po2(data_dict, rc, index=0):
    poly = data_dict['poly'][index]
    print(data_dict['image'][index].shape, data_dict['path'][index])

    # print(data_dict['poly'], '000')
    print(poly, 'b')
    print(data_dict['poly_meta'][index])

    # print(data_dict['ocrdet_kernel'].shape)
    # exit()

    res = rc(image=data_dict['image'][index], ori_size=data_dict['ori_size'][index], ocrdet_kernel=data_dict['ocrdet_kernel'][index])
    img = res['image']
    poly = res['poly']

    print(data_dict['ori_size'][index])
    print(poly, 'a')

    img = draw_polygon_without_label(img, poly)

    plt.figure()
    # plt.subplot(121)
    plt.imshow(img)
    plt.axis('off')


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    # torch.set_printoptions(precision=4, threshold=None, edgeitems=None, linewidth=None, profile=None)

    # cfg = load_config('spnx')
    cfg = load_config_far_away('configs/poly.py')
    data_manager = DataManager(cfg.test_data)
    dataloader = data_manager.load_data()
    info = data_manager.info

    time.sleep(0.1)

    rc = data_manager.oobmab
    print(data_manager)
    print(data_manager.dataset.bamboo.rper())
    # exit()

    for data in tqdm(dataloader):
        po(data, rc, 0)
        plt.show()
        break
