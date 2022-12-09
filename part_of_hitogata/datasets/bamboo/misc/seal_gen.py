import math
import torch
import numpy as np
from ..builder import INTERNODE
from ..base_internode import BaseInternode
from ...utils.common import get_image_size
from PIL import Image, ImageDraw, ImageFont


__all__ = ['SealGen']


@INTERNODE.register_module()
class SealGen(BaseInternode):
    def __init__(
        self, 
        font_path, 
        size=(512, 512), 
        ch_size=50, 
        ch_dist=70, 
        ch_angle_range=220, 
        star_ratio=0.7, 
        thickness=3s
    ):
        self.font_path = font_path
        self.size = size
        self.ch_size = ch_size
        self.ch_dist = ch_dist
        self.ch_angle_range = ch_angle_range
        self.star_ratio = star_ratio
        self.thickness = thickness

        self.font = 

    def __call__(self, data_dict):
        w, h = get_image_size(data_dict['image'])
        heatmaps_per_img = []
        visible_per_img = []

        print(data_dict['point'].shape)

        if 'point_meta' in data_dict.keys():
            visible = data_dict['point_meta']['visible']
        else:
            visible = np.empty(shape=data_dict['point'].shape[:2], dtype=np.bool)
            visible.fill(True)

        for points, vises in zip(data_dict['point'], visible):
            for point, vis in zip(points, vises):
                # print('a', vis)
                heatmap, new_vis = self.calc_heatmap((w, h), point, vis)
                # print(new_vis)
                heatmaps_per_img.append(heatmap.unsqueeze(0))
                visible_per_img.append(new_vis)

        visible_per_img = np.array(visible_per_img).reshape(visible.shape)
        data_dict['heatmap'] = torch.cat(heatmaps_per_img, dim=0)

        if 'point_meta' in data_dict.keys():
            data_dict['point_meta']['visible'] = visible_per_img

        return data_dict

    def __repr__(self):
        return 'SealGen()'.format()

