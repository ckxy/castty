import os
import numpy as np
from addict import Dict
from .reader import Reader
from .utils import read_image_paths
from .builder import READER


__all__ = ['ImageReader']


@READER.register_module()
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
        img = self.read_image(self.image_paths[index])
        w, h = img.size

        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=self.image_paths[index]
        )

    def __repr__(self):
        return 'ImageReader(root={}, {})'.format(self.root, super(ImageReader, self).__repr__())
