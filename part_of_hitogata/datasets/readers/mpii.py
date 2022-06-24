import os
import numpy as np
from addict import Dict
from PIL import Image
from .reader import Reader
from .builder import READER


__all__ = ['MPIIH5Reader']


@READER.register_module()
class MPIIH5Reader(Reader):
    def __init__(self, root, set_path, length=200, **kwargs):
        super(Market1501AttritubesReader, self).__init__(**kwargs)

        self.root = root
        self.img_root = os.path.join(self.root, 'images')
        
        assert os.path.exists(self.img_root)
        assert os.path.exists(set_path)

        self.set_path = set_path

        import h5py
        self.h5f = h5py.File(self.set_path, 'r')
        self.length = length

        self.data_lines = [0] * len(self.h5f['index'])

        assert len(self.data_lines) > 0

    def get_dataset_info(self):
        return range(len(self.data_lines)), Dict({})

    def get_data_info(self, index):
        return

    def __call__(self, index):
        # index = data_dict

        res = dict()

        img_path = os.path.join(self.img_root, self.h5f['imgname'][index].decode('UTF-8')) 
        # img = Image.open(img_path).convert('RGB')
        img = self.read_image(img_path)
        
        res['point'] = self.h5f['part'][index] - 1
        res['visible'] = self.h5f['visible'][index].astype(np.int)

        s = self.h5f['scale'][index]
        c = self.h5f['center'][index] - 1
        l = round(s * self.length / 2)
        x1, y1 = c[0] - l, c[1] - l
        x2, y2 = c[0] + l, c[1] + l
        res['bbox'] = np.array([[x1, y1, x2, y2]]).astype(np.float)
        res['mpii_scale'] = s

        res['image'] = img
        res['path'] = img_path
        w, h = img.size
        res['ori_size'] = np.array([h, w]).astype(np.float32)
        res['mpii_length'] = self.length

        return res

    def __repr__(self):
        return 'MPIIH5Reader(root={}, set_path={}, length={}, {})'.format(self.root, self.set_path, self.length, super(MPIIH5Reader, self).__repr__())
