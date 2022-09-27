import os
import json
import ntpath
import numpy as np
from addict import Dict
from PIL import Image, ImageDraw
from .reader import Reader
from .builder import READER


__all__ = ['LabelmeMaskReader']


# @READER.register_module()
class LabelmeMaskReader(Reader):
    def __init__(self, root, classes, **kwargs):
        super(LabelmeMaskReader, self).__init__(**kwargs)

        self.root = root
        self.classes = classes

        self.img_root = os.path.join(self.root, 'img')
        mask_root = os.path.join(self.root, 'json')

        assert os.path.exists(self.img_root)
        assert os.path.exists(mask_root)
        assert self.classes[0] == '__background__'

        self.mask_paths = sorted(os.listdir(mask_root))
        self.mask_paths = [os.path.join(mask_root, path) for path in self.mask_paths]

        assert len(self.mask_paths) > 0

    def get_dataset_info(self):
        return range(len(self.mask_paths)), Dict({'classes': self.classes})

    def get_data_info(self, index):
        with open(self.mask_paths[index], 'r') as f:
            load_dict = json.load(f)
        w, h = load_dict['imageWidth'], load_dict['imageHeight']
        return dict(h=h, w=w)

    def __call__(self, index):
        # index = data_dict

        with open(self.mask_paths[index], 'r') as f:
            load_dict = json.load(f)

        w, h = load_dict['imageWidth'], load_dict['imageHeight']
        mask = Image.new('P', (w, h), 0)

        for s in load_dict['shapes']:
            if s['shape_type'] != 'polygon':
                continue
            if s['label'] not in self.classes:
                continue
            # print(pts)
            # print(s['label'], self.classes.index(s['label']))
            pts = np.array(s['points']).astype(np.int).flatten().tolist()
            ImageDraw.Draw(mask).polygon(pts, fill=self.classes.index(s['label']))

        path = os.path.join(self.img_root, ntpath.basename(load_dict['imagePath']))

        # img = Image.open(path).convert('RGB')
        img = self.read_image(path)
        # return {'image': img, 'ori_size': np.array([h, w]).astype(np.float32), 'path': path, 'mask': mask}
        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=path,
            mask=mask
        )

    def __repr__(self):
        return 'LabelmeMaskReader(root={}, classes={}, {})'.format(self.root, self.classes, super(LabelmeMaskReader, self).__repr__())
