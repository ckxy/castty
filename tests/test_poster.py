import os
import cv2
import math
import time
import torch
import ntpath
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from castty.datasets import DataManager
from castty.config import load_config, load_config_far_away
from castty.utils.bbox_tools import draw_bbox, draw_bbox_without_label


def ga(data_dict, classes, rc, index=0):
    bboxes = data_dict['bbox'][index].cpu().numpy()
    # print(data_dict['image'][index].shape)
    # print(data_dict['ds_label'].shape)
    # print(data_dict['ds_image'].shape)

    print(bboxes, bboxes.shape, 'b')

    ori_size = data_dict['image_meta'][index]['ori_size']

    # res = rc(image=data_dict['image'][index], remove=data_dict['remove'][index], ori_size=ori_size, bbox=bboxes)
    res = rc(image=data_dict['image'][index], ori_size=ori_size, bbox=bboxes)

    # b_img = draw_bbox(res['image'].copy(), res['bbox'], data_dict['bbox_meta'][index].get('class_id', None), classes, data_dict['bbox_meta'][index].get('score', None))
    b_img = draw_bbox_without_label(res['image'].copy(), res['bbox'])
    # c_img = draw_bbox_without_label(res['remove'].copy(), res['bbox'])

    # print(bboxes, 'a')
    # print(ori_size)

    # plt.subplot(121)
    plt.imshow(b_img)
    plt.axis('off')
    # plt.subplot(122)
    # plt.imshow(c_img)
    # plt.axis('off')


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    # torch.set_printoptions(precision=4, threshold=None, edgeitems=None, linewidth=None, profile=None)

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = load_config_far_away('configs/poster.py')
    data_manager = DataManager(cfg.test_data)
    dataloader = data_manager.load_data()
    info = data_manager.info

    time.sleep(0.1)

    rc = data_manager.oobmab
    print(data_manager)
    print(data_manager.dataset.bamboo.rper())
    # exit()

    # data_manager.dataset[0]
    # exit()

    for data in tqdm(dataloader):
        # pass
        ga(data, info['bbox']['classes'], rc, 0)
        plt.show()
        break
