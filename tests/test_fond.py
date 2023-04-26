import os
import math
import time
import torch
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from part_of_hitogata.datasets import DataManager
from part_of_hitogata.configs import load_config, load_config_far_away
from part_of_hitogata.utils.bbox_tools import draw_bbox
from part_of_hitogata.utils.point_tools import draw_point
from part_of_hitogata.utils.mask_tools import draw_mask
from part_of_hitogata.utils.polygon_tools import draw_polygon
from part_of_hitogata.utils.label_tools import draw_label


def kp(data_dict, rc, data_info, index=0):
    print(data_dict.keys())
    print(data_dict['image'][index].shape)

    ori_size = data_dict['image_meta'][0]['ori_size']

    res = dict(image=data_dict['image'][index], ori_size=ori_size)

    if 'bbox' in data_dict.keys():
        res['bbox'] = data_dict['bbox'][index].numpy()
    if 'mask' in data_dict.keys():
        res['mask'] = data_dict['mask'][index]
    if 'point' in data_dict.keys():
        res['point'] = data_dict['point'][index].numpy()
    if 'poly' in data_dict.keys():
        res['poly'] = data_dict['poly'][index]

    res = rc(**res)
    img = res['image']

    if 'mask' in data_dict.keys():
        img, _ = draw_mask(img, res['mask'], data_info['mask']['classes'])
    if 'point' in data_dict.keys():
        img = draw_point(img, res['point'], data_dict['point_meta'][index].get('keep', None))
    if 'bbox' in data_dict.keys():
        img = draw_bbox(img, res['bbox'], data_dict['bbox_meta'][index].get('class_id', None), data_info['bbox']['classes'], data_dict['bbox_meta'][index].get('score', None))
    if 'poly' in data_dict.keys():
        img = draw_polygon(img, res['poly'], data_dict['poly_meta'][index].get('keep', None), data_dict['poly_meta'][index].get('class_id', None), data_info['poly']['classes'])
    
    plt.imshow(img)
    # plt.axis('off')


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    # torch.set_printoptions(precision=4, threshold=None, edgeitems=None, linewidth=None, profile=None)

    cfg = load_config_far_away('configs/base.py')
    data_manager = DataManager(cfg.test_data)
    dataloader = data_manager.load_data()
    info = data_manager.info

    time.sleep(0.1)

    rc = data_manager.oobmab
    print(data_manager)
    print(data_manager.dataset.bamboo.rper())
    # exit()

    for data in tqdm(dataloader):
        kp(data, rc, info, 0)
        # hm(data, rc, 0)
        plt.show()
        break


