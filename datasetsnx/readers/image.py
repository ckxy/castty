import os
import numpy as np
from addict import Dict
from PIL import Image
from .reader import Reader
from .utils import read_image_paths


__all__ = ['ImageReader']


class ImageReader(Reader):
    def __init__(self, root, **kwargs):
        super(ImageReader, self).__init__(**kwargs)

        assert os.path.exists(root)
        self.root = root
        self.image_paths = read_image_paths(self.root)
        assert len(self.image_paths) > 0

    def get_dataset_info(self):
        return range(len(self.image_paths)), Dict({})

    def get_data_info(self, index):
        return

    def __call__(self, index):
        # index = data_dict
        # img = Image.open(self.image_paths[index]).convert('RGB')
        img = self.read_image(self.image_paths[index])
        w, h = img.size
        path = self.image_paths[index]
        # return {'image': img, 'ori_size': np.array([h, w]).astype(np.float32), 'path': path}
        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=path
        )

    def __repr__(self):
        return 'ImageReader(root={}, {})'.format(self.root, super(ImageReader, self).__repr__())
