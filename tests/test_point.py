import os
import cv2
import math
import time
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from castty.datasets import DataManager
from castty.config import load_config, load_config_far_away
from castty.utils.point_tools import draw_point
from castty.utils.heatmap_tools import draw_heatmap
from castty.utils.bbox_tools import draw_bbox


# def kp(data_dict, rc, index=0):
#     print(data_dict['image'][index].shape)
#     bboxes = data_dict['bbox'][index].cpu().numpy()
#     points = data_dict['point'][index].numpy()

#     ori_size = data_dict['image_meta'][index]['ori_size']

#     res = rc(image=data_dict['image'][index], ori_size=ori_size, point=points, bbox=bboxes)
#     img = res['image']
#     points = res['point']
#     bboxes = res['bbox']

#     print(os.path.splitext(os.path.basename(data_dict['image_meta'][index]['path']))[0])
#     print(data_dict['image_meta'][index]['path'])
#     # print(data_dict['euler_angle'][index])

#     img = draw_point(img, points, data_dict['point_meta'][index].get('keep', None))
#     img = draw_bbox(img, bboxes)
#     plt.imshow(img)
#     plt.axis('off')


def kp(data_dict, rc, index=0):
    print(data_dict['image'][index].shape)
    points = data_dict['point'][index].numpy()

    ori_size = data_dict['image_meta'][index]['ori_size']

    res = rc(image=data_dict['image'][index], ori_size=ori_size, point=points)
    img = res['image']
    points = res['point']

    print(os.path.splitext(os.path.basename(data_dict['image_meta'][index]['path']))[0])
    print(data_dict['image_meta'][index]['path'])
    # print(data_dict['euler_angle'][index])

    img = draw_point(img, points, data_dict['point_meta'][index].get('keep', None))
    plt.imshow(img)
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

