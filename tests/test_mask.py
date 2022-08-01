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
from part_of_hitogata.utils.mask_tools import draw_mask


def ss(data_dict, classes, rc, index=0):
    img = data_dict['image'][index]
    mask = data_dict['mask'][index]

    print(img.shape, mask.shape, data_dict['ori_size'][index])

    res = rc(image=img, ori_size=data_dict['ori_size'][index], mask=mask)
    img = res['image']
    mask = res['mask']

    plt.subplot(131)
    plt.imshow(img)
    # plt.axis('off')
    plt.subplot(132)
    mask, bar = draw_mask(img, mask, classes)
    plt.imshow(mask)
    plt.subplot(133)
    plt.imshow(bar)
    plt.axis('off')


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    # torch.set_printoptions(precision=4, threshold=None, edgeitems=None, linewidth=None, profile=None)

    cfg = load_config_far_away('configs/mask.py')
    data_manager = DataManager(cfg.test_data)
    dataloader = data_manager.load_data()
    info = data_manager.info

    time.sleep(0.1)

    rc = data_manager.oobmab
    print(data_manager)
    print(data_manager.dataset.bamboo.rper())
    # exit()

    for data in tqdm(dataloader):
        ss(data, info['mask_classes'], rc, 0)
        plt.show()
        break

