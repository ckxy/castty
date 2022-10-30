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
from part_of_hitogata.utils.seq_tools import draw_seq


def sq(data_dict, rc, index=0):
    img = data_dict['image'][index]
    seq = data_dict['seq'][index]

    print(img.shape)

    res = rc(image=img, ori_size=data_dict['ori_size'][index])
    img = res['image']

    img = draw_seq(img, seq)

    plt.imshow(img)
    plt.axis('off')


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    # torch.set_printoptions(precision=4, threshold=None, edgeitems=None, linewidth=None, profile=None)

    cfg = load_config_far_away('configs/seq.py')
    data_manager = DataManager(cfg.test_data)
    dataloader = data_manager.load_data()
    info = data_manager.info

    time.sleep(0.1)

    rc = data_manager.oobmab
    print(data_manager)
    print(data_manager.dataset.bamboo.rper())

    for data in tqdm(dataloader):
        # print(data['encoded_seq'], data['encoded_seq'].shape)
        sq(data, rc, 0)
        plt.show()
        # break

