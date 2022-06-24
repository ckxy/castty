import os
import cv2
import math
import time
import torch
import ntpath
from PIL import Image
import numpy as np
from configs import load_config, load_config_far_away
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.bbox_tools import xywh2xyxy, xyxy2xywh, draw_bbox, grid_analysis
from utils.point_tools import draw_point
from utils.mask_tools import draw_mask
from utils.label_tools import draw_label, draw_grouped_label
from utils.seq_tools import draw_seq, get_seq_from_batch, seq2str
from utils.heatmap_tools import draw_heatmap
from utils.polygon_tools import draw_polygon_without_label

from datasetsnx import create_data_manager


def ga(data_dict, classes, rc, index=0):
    bboxes = data_dict['bbox'][index].cpu().numpy()
    print(data_dict['image'][index].shape)

    print(data_dict['bbox'][0].numpy(), 'b')

    res = rc(image=data_dict['image'][index], ori_size=data_dict['ori_size'][index], bbox=bboxes, training=True)
    img1 = res['image']
    img2 = img1.copy()
    bboxes = res['bbox']

    # print(len(data_dict['ga_bbox'][index]), len(data_dict['ga_index'][index]))

    if 'ga_bbox' in data_dict:
        ga_img = grid_analysis(img1, (8, 16, 32), data_dict['ga_bbox'][index], data_dict['ga_index'][index], len(bboxes))

    a = bboxes[:, :4]
    b = bboxes[:, 4:5]
    c = bboxes[:, 5:]
    d = np.concatenate((a, c, b), axis=1)

    b_img = draw_bbox(img2, d, classes)
    # b_img.save('1.jpg')
    # ga_img.save('atssa.jpg')

    print(b_img, data_dict['ori_size'][index])

    if 'ga_bbox' in data_dict:
        plt.subplot(211)
        plt.imshow(b_img)
        plt.axis('off')
        plt.subplot(212)
        plt.imshow(ga_img)
        plt.axis('off')
    else:
        plt.imshow(b_img)
        plt.axis('off')

    # print(data_dict['neg_category_ids'][index])
    # print(data_dict['not_exhaustive_category_ids'][index])
    # neg_category_ids = [classes[i] for i in data_dict['neg_category_ids'][index]]
    # not_exhaustive_category_ids = [classes[i] for i in data_dict['not_exhaustive_category_ids'][index]]
    # print(neg_category_ids)
    # print(not_exhaustive_category_ids)


def im(data_dict, rc, index=0):
    res = rc(image=data_dict['image'][index], jump=['Resize', 'Padding'])
    img = res['image']

    print(img.size[::-1], os.path.splitext(ntpath.basename(data_dict['path'][index]))[0])
    plt.imshow(img)
    plt.axis('off')
    img.save('/home/ubuntu/test/dolls/544/{}'.format(ntpath.basename(data_dict['path'][index])))


def kp(data_dict, rc, index=0):
    print(data_dict['image'][index].shape)

    res = rc(image=data_dict['image'][index], ori_size=data_dict['ori_size'][index], point=data_dict['point'][index], training=True)
    img = res['image']
    points = res['point'].numpy()

    print(img.size[::-1], os.path.splitext(ntpath.basename(data_dict['path'][index]))[0])
    print(data_dict['path'][index])
    # print(data_dict['euler_angle'][index])

    img = draw_point(img, points)
    plt.imshow(img)
    plt.axis('off')


def hm(data_dict, rc, index=0):
    heatmaps = data_dict['heatmap'][index]

    res = rc(image=data_dict['image'][index], ori_size=data_dict['ori_size'][index], heatmap=heatmaps, training=True)
    img = res['image']
    # p = res['point']
    heatmaps = res['heatmap']
    # heatmaps = heatmaps.unsqueeze(0)
    print(heatmaps.shape)
    # exit()
    # print(data_dict['point'][index])
    # print(data_dict['point'][index] - p)
    # print((data_dict['point'][index] - p).mean())
    # exit()

    # cols = math.ceil((1 + len(heatmaps)) / 4)
    # plt.subplot(3, 1, 1)
    # plt.imshow(img)
    # plt.axis('off') 

    # for i, heatmap in enumerate(heatmaps):
    #     tmp = draw_heatmap(heatmap)
    #     tmp = tmp.resize(img.size, Image.BILINEAR)

    #     res = Image.blend(img, tmp, 0.5)
    #     plt.subplot(3, 1, i + 2)
    #     plt.imshow(res)
    #     plt.axis('off') 

    for i, heatmap in enumerate(heatmaps):
        tmp = draw_heatmap(heatmap)
        # tmp.save('{}.jpg'.format(i))
        print(img.size, tmp.size)

        res = Image.blend(img, tmp, 0.5)
        plt.subplot(4, 4, i + 1)
        plt.imshow(res)
        plt.axis('off')


def cr(data_dict, rc, index=0):
    print(data_dict['image'][index].shape)
    # heatmaps = data_dict['heatmap'][index]
    h, w = data_dict['ori_size'][index].numpy()
    # rh, rw = 512.0 / h, 512.0 / w

    res = rc(image=data_dict['image'][index], ori_size=data_dict['ori_size'][index], training=True)
    img = res['image']

    res = rc(ori_size=data_dict['ori_size'][index], heatmap_char=data_dict['heatmap_char'][index], training=True)
    tmp = draw_heatmap(res['heatmap'])
    print(img.size, tmp.size)
    res = Image.blend(img, tmp, 0.5)
    plt.subplot(2, 1, 1)
    plt.imshow(res)
    plt.axis('off')

    res = rc(ori_size=data_dict['ori_size'][index], heatmap_affinity=data_dict['heatmap_affinity'][index], training=True)
    tmp = draw_heatmap(res['heatmap'])
    print(img.size, tmp.size)
    res = Image.blend(img, tmp, 0.5)
    plt.subplot(2, 1, 2)
    plt.imshow(res)
    plt.axis('off')


def cm(data_dict, rc):
    # l = data_dict['b_label'][0]
    # print(l, rc(b_label=l)['b_label'])
    # exit()
    a = []
    for aimg in data_dict['a_image']:
        a.append(np.array(rc(a_image=aimg)['a_image']))
    ast = []
    for astimg in data_dict['a_star_image']:
        ast.append(np.array(rc(a_star_image=astimg)['a_star_image']))
    b = []
    for bimg in data_dict['b_image']:
        b.append(np.array(rc(b_image=bimg)['b_image']))

    for i in range(len(a)):
        plt.subplot(3, 6, 3 * i + 1)
        plt.title('a')
        plt.imshow(a[i])
        plt.axis('off')

        plt.subplot(3, 6, 3 * i + 2)
        plt.title('ast')
        plt.imshow(ast[i])
        plt.axis('off')

        plt.subplot(3, 6, 3 * i + 3)
        plt.title('b')
        plt.imshow(b[i])
        plt.axis('off')


def mst(data_dict, classes, rc, s=0, index=0):
    # bboxes = data_dict['bbox'][data_dict['bbox'][:, 0] == index][:, 1:]
    bboxes = data_dict['bbox'][index]

    img = data_dict['image'][s][index]
    print(img.shape)
    print(bboxes.numpy())

    res = rc(image=img, ori_size=data_dict['ori_size'][index], bbox=bboxes)
    img2 = res['image']
    bboxes = res['bbox'].numpy()

    a = bboxes[:, :4]
    b = bboxes[:, 4:5]
    c = bboxes[:, 5:]
    d = np.concatenate((a, c, b), axis=1)
    b_img = draw_bbox(img2, d, classes)

    plt.imshow(b_img)
    plt.axis('off')


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


def cl(data_dict, classes, rc, index=0):
    img = data_dict['image'][index]
    label = data_dict['label'][index].item()

    print(img.shape)
    print(data_dict['path'][index])
    print(label)

    res = rc(image=img, ori_size=data_dict['ori_size'][index])
    img = res['image']

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


def sq(data_dict, chars, rc, index=0):
    img = data_dict['image'][index]
    seq = data_dict['seq'][index]
    # seq = get_seq_from_batch(data_dict['seq'], data_dict['seq_length'], 0)
    # seq = seq2str(seq, chars)

    # print(img.shape)
    # print(data_dict['path'][index])
    # print(seq)

    res = rc(image=img, ori_size=data_dict['ori_size'][index], jump=['Resize', 'Padding'])
    img = res['image']

    img = draw_seq(img, seq)

    plt.title(seq)
    plt.imshow(img)
    plt.axis('off')


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    # torch.set_printoptions(precision=4, threshold=None, edgeitems=None, linewidth=None, profile=None)

    # cfg = load_config_far_away('/media/ubuntu/PANFU/SanDisk/c/spnx.py')
    cfg = load_config('spnx')
    # cfg = load_config('nano416')

    # data_manager = create_data_manager(cfg.train_data)
    data_manager = create_data_manager(cfg.test_data)
    dataloader = data_manager.load_data()
    info = data_manager.info

    # print(dataloader.dataset.bamboo)

    # for data in tqdm(dataloader):
    #     break
    # exit()
    
    time.sleep(0.1)

    rc = info['oobmab']
    # print(rc)
    print(data_manager.dataset.bamboo.rper())
    # exit()

    for _ in range(1):
        for data in tqdm(dataloader):
            # continue
            # print(data.keys())
            # print(data['point'].shape, data['euler_angle'].shape)
            # p, e = torch.load('../dolls0828/a.pth')
            # # print((p == data['point']).all())
            # # print((e == data['euler_angle']).all())
            # # print(p)
            # # print(data['point'])
            # for i in range(len(p)):
            #     print(i, (p[i] == data['point'][i]).all())
            #     print(i, (e[i] == data['euler_angle'][i]).all())
            #     print(torch.abs(p[i] - data['point'][i]).mean())
            # exit()
            # break
            ga(data, dataloader.dataset.info.classes, rc, 0)
            # ss(data, dataloader.dataset.info.classes, rc, 0)
            # mst(data, dataloader.dataset.info.classes, rc, 0, 0)
            # im(data, rc, 0)
            # kp(data, rc, 0)
            # hm(data, rc, 0)
            # cr(data, rc, 0)
            # cm(data, rc)
            # cl(data, dataloader.dataset.info.classes, rc, 0)
            # mcl(data, dataloader.dataset.info.classes, rc, 0)
            # sq(data, dataloader.dataset.info.chars, rc, 0)
            plt.show()
            break
            # pass
            # tqdm.write(data['path'][0])
            # plt.pause(0.5)