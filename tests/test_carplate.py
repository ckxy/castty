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
from castty.utils.seq_tools import draw_seq
from castty.utils.point_tools import draw_point
from castty.utils.bbox_tools import draw_bbox


def sq(data_dict, rc, info, index=0):
    img = data_dict['image'][index]
    seq = data_dict['seq'][index]

    bboxes = data_dict['bbox'][index].cpu().numpy()
    points = data_dict['point'][index].numpy()
    ori_size = data_dict['image_meta'][index]['ori_size']

    print(data_dict.keys())
    print(img.shape)

    res = rc(image=img, ori_size=ori_size, point=points, bbox=bboxes)
    img = res['image']

    img = draw_point(img, points, data_dict['point_meta'][index].get('keep', None))
    img = draw_bbox(img, res['bbox'], data_dict['bbox_meta'][index].get('class_id', None), info['bbox']['classes'], data_dict['bbox_meta'][index].get('score', None))
    img = draw_seq(img, seq)

    plt.imshow(img)
    plt.axis('off')


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    # torch.set_printoptions(precision=4, threshold=None, edgeitems=None, linewidth=None, profile=None)

    cfg = load_config_far_away('configs/carplate.py')
    data_manager = DataManager(cfg.test_data)
    dataloader = data_manager.load_data()
    info = data_manager.info

    time.sleep(0.1)

    rc = data_manager.oobmab
    print(data_manager)
    print(data_manager.dataset.bamboo.rper())

    for data in tqdm(dataloader):
        # print(data['encoded_seq'], data['encoded_seq'].shape)
        sq(data, rc, info, 0)
        plt.show()
        break

