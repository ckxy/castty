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
from part_of_hitogata.utils.label_tools import draw_label, draw_grouped_label


def cl(data_dict, classes, rc, index=0):
    img = data_dict['image'][index]
    label = data_dict['label'][index].numpy()

    print(img.shape)
    print(data_dict['path'][index])
    print(label, 'label')

    res = rc(image=img, ori_size=data_dict['ori_size'][index], label=label)
    img = res['image']
    label = res['label']

    img = draw_label(img, label, classes)

    plt.title(classes[label])
    plt.imshow(img)
    plt.axis('off')


def mcl(data_dict, classes, rc, index=0):
    img = data_dict['image'][index]
    label = data_dict['label'][index].numpy()

    print(img.shape)
    print(data_dict['path'][index])
    # for i in range(len(classes)):
    #     print(i, classes[i][label[i]])
    # exit()

    # print(img.shape)

    res = rc(image=img, ori_size=data_dict['ori_size'][index], training=True)
    img = res['image']
    # print(img.size)

    # img = img.resize((img.size[0] * 8, img.size[1] * 8), Image.BILINEAR)
    img = draw_grouped_label(img, label, classes)

    # plt.title(classes[label])
    plt.imshow(img)
    plt.axis('off')


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    # torch.set_printoptions(precision=4, threshold=None, edgeitems=None, linewidth=None, profile=None)

    cfg = load_config_far_away('configs/label.py')
    data_manager = DataManager(cfg.test_data)
    dataloader = data_manager.load_data()
    info = data_manager.info

    time.sleep(0.1)

    rc = data_manager.oobmab
    print(data_manager)
    print(data_manager.dataset.bamboo.rper())
    # exit()

    for data in tqdm(dataloader):
        cl(data, info['classes'], rc, 0)
        plt.show()
        break

