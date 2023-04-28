import os
import json
import numpy as np
from .reader import Reader
from .builder import READER
from PIL import Image, ImageDraw


__all__ = ['LabelmeMaskReader']


@READER.register_module()
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

        self._info = dict(
            forcat=dict(
                mask=dict(
                    classes=self.classes
                )
            )
        )

    def __getitem__(self, index):
        with open(self.mask_paths[index], 'r') as f:
            load_dict = json.load(f)

        w, h = load_dict['imageWidth'], load_dict['imageHeight']
        mask = Image.new('P', (w, h), 0)

        for s in load_dict['shapes']:
            if s['shape_type'] != 'polygon':
                continue
            if s['label'] not in self.classes:
                continue
            pts = np.array(s['points']).astype(np.int32).flatten().tolist()
            ImageDraw.Draw(mask).polygon(pts, fill=self.classes.index(s['label']))
            
        mask = np.array(mask).astype(np.int32)

        path = os.path.join(self.img_root, os.path.basename(load_dict['imagePath']))
        img = self.read_image(path)

        return dict(
            image=img,
            image_meta=dict(ori_size=(w, h), path=path),
            mask=mask
        )

    def __len__(self):
        return len(self.mask_paths)

    def __repr__(self):
        return 'LabelmeMaskReader(root={}, classes={}, {})'.format(self.root, self.classes, super(LabelmeMaskReader, self).__repr__())
