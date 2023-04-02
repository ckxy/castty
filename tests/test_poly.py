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
from part_of_hitogata.configs import load_config, load_config_far_away
from part_of_hitogata.utils.polygon_tools import draw_polygon, get_cw_order_form


def cross(p1, p2, p3):#跨立实验
    x1 = p2[0] - p1[0]
    y1 = p2[1] - p1[1]
    x2 = p3[0] - p1[0]
    y2 = p3[1] - p1[1]
    return x1 * y2 - x2 * y1     

def isintersec(p1, p2, p3, p4): #判断两线段是否相交
    #快速排斥，以l1、l2为对角线的矩形必相交，否则两线段不相交
    if max(p1[0], p2[0]) >= min(p3[0], p4[0]) and max(p3[0], p4[0]) >= min(p1[0], p2[0]) and max(p1[1], p2[1]) >= min(p3[1], p4[1]) and max(p3[1], p4[1]) >= min(p1[1], p2[1]):
        #若通过快速排斥则进行跨立实验
        if cross(p1, p2, p3) * cross(p1, p2, p4) <= 0 and cross(p3, p4, p1) * cross(p3, p4, p2) <= 0:
            res = True
        else:
            res = False
    else:
        res = False
    return res


def po3(data_dict, rc, index=0):
    polys = data_dict['poly'][index]
    print(data_dict['image'][index].shape, data_dict['path'][index])

    print(data_dict['poly_meta'][index])

    res = rc(image=data_dict['image'][index], ori_size=data_dict['ori_size'][index], poly=polys)
    img = res['image']
    polys = res['poly']

    # img = draw_polygon(img, polys)

    polys = [get_cw_order_form(poly) for poly in polys]
    print(polys, 'a')
    centers = [np.mean(x, axis=0) for x in polys]
    draw = ImageDraw.Draw(img)

    res_poly = polys[0].copy()
    for i in range(len(centers) - 1):
        c1 = centers[i].astype(np.int32).tolist()
        c2 = centers[i + 1].astype(np.int32).tolist()
        # draw.polygon(c1 + c2, outline=(255, 0, 40), width=1)

        for j1 in range(len(res_poly)):
            p1 = res_poly[j1]
            p2 = res_poly[(j1 + 1) % len(res_poly)]

            if isintersec(p1, p2, c1, c2):
                break

        poly1 = np.roll(res_poly, len(res_poly) - j1, 0)
        # print(poly1)

        for j2 in range(len(polys[i + 1])):
            p1 = polys[i + 1][j2]
            p2 = polys[i + 1][(j2 + 1) % len(polys[i + 1])]

            if isintersec(p1, p2, c1, c2):
                break

        poly2 = np.roll(polys[i + 1], len(polys[i + 1]) - j2, 0)
        # print(poly2)

        poly1 = np.roll(poly1, len(poly1) - 1, 0)
        poly2 = np.roll(poly2, len(poly2) - 1, 0)
        res_poly = np.concatenate([poly1, poly2], axis=0)
        # print(res_poly)
        # print('b')
        # exit()
        # break
    print(res_poly)
    draw.polygon(res_poly.astype(np.int32).flatten().tolist(), outline=(255, 0, 255), width=2)

    plt.figure()
    # plt.subplot(121)
    plt.imshow(img)
    plt.axis('off')

    return


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

    img = draw_polygon(img, poly)

    plt.figure()
    # plt.subplot(121)
    plt.imshow(img)
    plt.axis('off')


def po(data_dict, rc, index=0):
    poly = data_dict['poly'][index]
    print(data_dict['image'][index].shape, data_dict['path'][index])

    # print(data_dict['poly'], '000')
    # print(poly, 'b')
    print(data_dict['poly_meta'][index])

    # a = np.array([[260, 497], [365, 497], [365, 588], [260, 588]]).astype(np.float32)
    # poly[-1] = a

    res = rc(image=data_dict['image'][index], ori_size=data_dict['ori_size'][index], poly=poly)
    img = res['image']
    poly = res['poly']

    # from part_of_hitogata.utils.mask_tools import draw_mask
    # mask = data_dict['mask'][index]
    # res = rc(ori_size=data_dict['ori_size'][index], mask=mask)
    # mask = res['mask']
    # img, bar = draw_mask(img, mask, ['bgd', 'fgd'])

    print(data_dict['ori_size'][index])
    # print(poly, 'a')

    # img = draw_polygon(img, poly, data_dict['poly_meta'][index]['ignore_flag'])
    img = draw_polygon(img, poly, data_dict['poly_meta'][index].get('ignore_flag', None), data_dict['poly_meta'][index].get('class_id', None), [0,1])

    # print(data_dict['ocrdet_kernel'].shape, data_dict['ocrdet_train_mask'].shape)

    plt.figure()
    # plt.subplot(121)
    plt.imshow(img)
    plt.axis('off')

    return

    # plt.subplot(122)
    # plt.imshow(data_dict['ocrdet_train_mask'][index].numpy(), cmap='gray')
    # plt.axis('off')

    # plt.figure()
    # for i in range(len(data_dict['ocrdet_kernel'][index])):
    #     plt.subplot(len(data_dict['ocrdet_kernel'][index]) // 3, 3, i + 1)
    #     plt.imshow(data_dict['ocrdet_kernel'][index][i].numpy(), cmap='gray')
    #     plt.axis('off')

    print(data_dict['ocrdet_shrink_map'].shape, data_dict['ocrdet_shrink_mask'].shape)
    print(data_dict['ocrdet_shrink_map'].dtype, data_dict['ocrdet_shrink_mask'].dtype)
    print(data_dict['ocrdet_thr_map'].shape, data_dict['ocrdet_thr_mask'].shape)
    print(data_dict['ocrdet_thr_map'].dtype, data_dict['ocrdet_thr_mask'].dtype)
    # exit()

    plt.subplot(2, 3, 1)
    plt.title('img')
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title('shrink_mask')
    plt.imshow(data_dict['ocrdet_shrink_mask'][index][0].numpy(), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title('thr_mask')
    plt.imshow(data_dict['ocrdet_thr_mask'][index][0].numpy(), cmap='gray')
    plt.axis('off')

    # plt.subplot(2, 3, 4)
    # plt.title('shrink_map')
    # plt.imshow(data_dict['ocrdet_shrink_map'][index][1].numpy(), cmap='gray')
    # plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title('shrink_map')
    plt.imshow(data_dict['ocrdet_shrink_map'][index][0].numpy(), cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title('thr_map')
    plt.imshow(data_dict['ocrdet_thr_map'][index][0].numpy(), cmap='gray')
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

