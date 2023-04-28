import os
import cv2
import math
import time
import torch
import ntpath
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from part_of_hitogata.datasets import DataManager
from part_of_hitogata.config import load_config, load_config_far_away
from part_of_hitogata.utils.polygon_tools import draw_polygon, get_cw_order_form


def po(data_dict, rc, index=0):
    poly = data_dict['poly'][index]
    print(data_dict['image'][index].shape, data_dict['image_meta'][index]['path'])

    # print(data_dict['poly'], '000')
    # print(poly, 'b')
    print(data_dict['poly_meta'][index])

    # a = np.array([[260, 497], [365, 497], [365, 588], [260, 588]]).astype(np.float32)
    # poly[-1] = a

    ori_size = data_dict['image_meta'][index]['ori_size']

    res = rc(image=data_dict['image'][index], ori_size=ori_size, poly=poly)
    img = res['image']
    poly = res['poly']

    # from part_of_hitogata.utils.mask_tools import draw_mask
    # mask = data_dict['mask'][index]
    # res = rc(ori_size=ori_size, mask=mask)
    # mask = res['mask']
    # img, bar = draw_mask(img, mask, ['bgd', 'fgd'])

    print(ori_size)
    # print(poly, 'a')

    # img = draw_polygon(img, poly, data_dict['poly_meta'][index]['ignore_flag'])
    img = draw_polygon(img, poly, data_dict['poly_meta'][index].get('keep', None), data_dict['poly_meta'][index].get('class_id', None), [0,1])

    plt.figure()
    # plt.subplot(121)
    plt.imshow(img)
    plt.axis('off')

    return

    # plt.subplot(122)
    # plt.imshow(data_dict['pse_mask'][index].numpy(), cmap='gray')
    # plt.axis('off')

    # return 

    # plt.figure()
    # for i in range(len(data_dict['pse_kernel'][index])):
    #     plt.subplot(len(data_dict['pse_kernel'][index]) // 3, 3, i + 1)
    #     plt.imshow(data_dict['pse_kernel'][index][i].numpy(), cmap='gray')
    #     plt.axis('off')

    # return

    print(data_dict['db_shrink_map'].shape, data_dict['db_shrink_mask'].shape)
    # print(data_dict['db_shrink_map'].dtype, data_dict['db_shrink_mask'].dtype)
    print(data_dict['db_thr_map'].shape, data_dict['db_thr_mask'].shape)
    # print(data_dict['db_thr_map'].dtype, data_dict['db_thr_mask'].dtype)
    # exit()

    plt.subplot(2, 3, 1)
    plt.title('img')
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title('shrink_mask')
    plt.imshow(data_dict['db_shrink_mask'][index].numpy(), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title('thr_mask')
    plt.imshow(data_dict['db_thr_mask'][index].numpy(), cmap='gray')
    plt.axis('off')

    if len(data_dict['db_shrink_map'][index]) > 1:
        plt.subplot(2, 3, 4)
        plt.title('shrink_map')
        plt.imshow(data_dict['db_shrink_map'][index][1].numpy(), cmap='gray')
        plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title('shrink_map')
    plt.imshow(data_dict['db_shrink_map'][index][0].numpy(), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title('thr_map')
    plt.imshow(data_dict['db_thr_map'][index].numpy(), cmap='gray')
    plt.axis('off')

    plt.show()


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
        # pass
        po(data, rc, 0)
        plt.show()
        break

