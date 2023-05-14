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

from castty.datasets import DataManager
from castty.config import load_config, load_config_far_away
from castty.utils.label_tools import draw_label


def mcl(data_dict, rc, index=0):
    a_image = data_dict['a_image'][index]
    a_star_image = data_dict['a_star_image'][index]
    b_image = data_dict['b_image'][index]

    print(a_image.shape, a_star_image.shape, b_image.shape)

    res = rc(a_image=a_image, a_star_image=a_star_image, b_image=b_image)
    a_image = res['image']
    a_star_image = res['a_star_image']
    b_image = res['b_image']

    plt.subplot(221)
    # plt.title(classes[label])
    plt.imshow(a_image)
    plt.axis('off')

    plt.subplot(224)
    # plt.title(classes[label])
    plt.imshow(a_star_image)
    plt.axis('off')

    plt.subplot(222)
    # plt.title(classes[label])
    plt.imshow(b_image)
    plt.axis('off')


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    # torch.set_printoptions(precision=4, threshold=None, edgeitems=None, linewidth=None, profile=None)

    cfg = load_config_far_away('configs/cmte.py')
    data_manager = DataManager(cfg.test_data)
    dataloader = data_manager.load_data()
    info = data_manager.info

    time.sleep(0.1)

    rc = data_manager.oobmab
    print(data_manager)
    print(data_manager.dataset.bamboo.rper())
    # exit()

    for data in tqdm(dataloader):
        mcl(data, rc, 0)
        plt.show()
        break

