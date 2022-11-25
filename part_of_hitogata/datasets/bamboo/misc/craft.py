import cv2
import torch
import numpy as np
from ..builder import INTERNODE
from PIL import Image, ImageDraw
from ..base_internode import BaseInternode
from ...utils.common import get_image_size
# from torch.nn.functional import interpolate
# from ....utils.polygon_tools import get_cw_order_form


__all__ = ['CalcLinkMap']


@INTERNODE.register_module()
class CalcLinkMap(BaseInternode):
    def __init__(self, tag='poly', ratio=0.5):
        assert 0 < ratio < 1

        self.tag = tag
        self.ratio = ratio

        self.l1 = (0.5 - ratio / 2) / (0.5 + ratio / 2)
        self.l2 = (0.5 + ratio / 2) / (0.5 - ratio / 2)

    def forward(self, data_dict):
        # res_polys = []
        w, h = get_image_size(data_dict['image'])
        mask = Image.new('P', (w, h), 0)

        for poly in data_dict[self.tag]:
            if len(poly) % 2 != 0 or len(poly) < 2:
                continue

            pu = []
            pd = []

            for i in range(len(poly) // 2):
                a = poly[i]
                b = poly[len(poly) - 1 - i]

                x = (a[0] + self.l1 * b[0]) / (1 + self.l1)
                y = (a[1] + self.l1 * b[1]) / (1 + self.l1)
                pu.append([x, y])

                x = (a[0] + self.l2 * b[0]) / (1 + self.l2)
                y = (a[1] + self.l2 * b[1]) / (1 + self.l2)
                pd.append([x, y])

            # t = get_cw_order_form(np.array(pu + pd[::-1], dtype=np.float32))
            # res_polys.append(t)

            pts = np.array(pu + pd[::-1], dtype=np.int32).flatten().tolist()
            ImageDraw.Draw(mask).polygon(pts, fill=1)

        # data_dict['poly'] = res_polys
        data_dict['mask'] = np.asarray(mask)

        return data_dict

    def __repr__(self):
        return 'CalcLinkMap(tag={}, ratio={})'.format(self.tag, self.ratio)
