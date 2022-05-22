import os
import numpy as np
from addict import Dict
from PIL import Image
from .reader import Reader
from .utils import is_image_file
from .builder import READER

__all__ = ['ImageFolderReader']


@READER.register_module()
class ImageFolderReader(Reader):
    def __init__(self, root, **kwargs):
        super(ImageFolderReader, self).__init__(**kwargs)

        self.root = root
        self.classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        self.classes.sort()
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

        self.samples = []
        for target in sorted(self.class_to_idx.keys()):
            d = os.path.join(self.root, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_image_file(path):
                        item = (path, self.class_to_idx[target])
                        self.samples.append(item)

        assert len(self.samples) > 0

    def get_dataset_info(self):
        return range(len(self.samples)), Dict({'classes': self.classes})

    def get_data_info(self, index):
        img = Image.open(self.samples[index][0])
        w, h = img.size
        return dict(h=h, w=w)

    def __call__(self, index):
        # index = data_dict
        # img = Image.open(self.samples[index][0]).convert('RGB')
        img = self.read_image(self.samples[index][0])
        label = self.samples[index][1]
        w, h = img.size
        path = self.samples[index][0]
        # return {'image': img, 'ori_size': np.array([h, w]).astype(np.float32), 'path': path, 'label': label}
        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=path,
            label=label
        )

    def __repr__(self):
        return 'ImageFolderReader(root={}, classes={}, {})'.format(self.root, tuple(self.classes), super(ImageFolderReader, self).__repr__())
