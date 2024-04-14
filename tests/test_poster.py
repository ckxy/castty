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
from castty.utils.bbox_tools import xywh2xyxy, xyxy2xywh, draw_bbox, grid_analysis


def ga(data_dict, classes, rc, index=0):
    bboxes = data_dict['bbox'][index].cpu().numpy()
    print(data_dict['image'][index].shape)
    print(data_dict['ds_label'].shape)
    print(data_dict['ds_image'].shape)

    print(bboxes, 'b')

    ori_size = data_dict['image_meta'][index]['ori_size']

    res = rc(image=data_dict['image'][index], ori_size=ori_size, bbox=bboxes)
    img1 = res['image']
    img2 = img1.copy()
    bboxes = res['bbox']

    b_img = draw_bbox(img2, res['bbox'], data_dict['bbox_meta'][index].get('class_id', None), classes, data_dict['bbox_meta'][index].get('score', None))

    print(bboxes, 'a')
    print(ori_size)

    plt.imshow(b_img)
    plt.axis('off')


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    # torch.set_printoptions(precision=4, threshold=None, edgeitems=None, linewidth=None, profile=None)

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
        ga(data, info['bbox']['classes'], rc, 0)
        plt.show()
        break

