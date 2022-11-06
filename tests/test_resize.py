import os
import cv2
import math
import time
import torch
import ntpath
import random
from PIL import Image
import numpy as np
from tqdm import tqdm


class Resize(object):
    def __init__(self, size, keep_ratio=True, short=False, **kwargs):
        assert len(size) == 2
        assert size[0] > 0 and size[1] > 0

        self.size = size
        self.keep_ratio = keep_ratio
        self.short = short

    def calc_scale_and_new_size(self, w, h):
        tw, th = self.size
        rw, rh = tw / w, th / h

        if self.keep_ratio:
            if self.short:
                r = max(rh, rw)
                scale = (r, r)
            else:
                r = min(rh, rw)
                scale = (r, r)

            # print(r * w, r * h)
            # print(int(round(r * w)), int(round(r * h)))
            # new_size = (int(r * w), int(r * h))
            new_size = int(round(r * w)), int(round(r * h))
        else:
            scale = (rw, rh)
            new_size = (tw, th)

        return scale, new_size

    def __call__(self, ori_size):
        _, new_size = self.calc_scale_and_new_size(*ori_size)
        return new_size

    def reverse(self, new_size):
        if 'intl_resize_and_padding_reverse_flag' in data_dict.keys():
            h, w = data_dict['ori_size']
            h, w = int(h), int(w)
            scale, _ = self.calc_scale_and_new_size(w, h)
            data_dict['intl_scale'] = (1 / scale[0], 1 / scale[1])
            data_dict['intl_new_size'] = (w, h)
        return data_dict

    def __repr__(self):
        return 'Resize(size={}, keep_ratio={}, short={})'.format(self.size, self.keep_ratio, self.short)


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    GW, GH = 416, 416

    r = Resize((GW, GH), True, True)
    for i in tqdm(range(100000)):
        L = random.randint(10, 2000)
        t = random.random() * 3

        o = [0, 0]
        o[0] = max(int(L / math.sqrt(t)), 1)
        o[1] = max(int(L * math.sqrt(t)), 1)

        # o = [23, 15]

        s = r(o)
        # print(s, o)
        if s[0] != GW and s[1] != GH:
            print(o, s)
            # print(s[0], GW, s[1], GH)
            print('---------')
        # break
