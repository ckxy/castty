import os
import json
import numpy as np
from .reader import Reader
from .builder import READER
from scipy.io import loadmat


__all__ = ['MPIIReader']


@READER.register_module()
class MPIIReader(Reader):
    def __init__(self, root, set_path, length=200, **kwargs):
        self.root = root
        self.img_root = os.path.join(self.root, 'images')
        
        assert os.path.exists(self.img_root)
        assert os.path.exists(set_path)

        self.set_path = set_path

        mat = loadmat(set_path)
        print(mat['RELEASE'])
        exit()
        self.length = length

        self.data_lines = [0] * len(self.h5f['index'])

        assert len(self.data_lines) > 0

        self._info = dict(
            forcat=dict(
                type='kpt',
            )
        )

    def __call__(self, index):
        img_path = os.path.join(self.img_root, self.h5f['imgname'][index].decode('UTF-8')) 
        img = self.read_image(img_path)
        
        res['point'] = self.h5f['part'][index] - 1
        res['visible'] = self.h5f['visible'][index].astype(np.int32)

        print(self.h5f['part'][index] - 1)
        print(self.h5f['visible'][index].astype(np.int32))
        exit()

        s = self.h5f['scale'][index]
        c = self.h5f['center'][index] - 1
        l = round(s * self.length / 2)
        x1, y1 = c[0] - l, c[1] - l
        x2, y2 = c[0] + l, c[1] + l
        res['bbox'] = np.array([[x1, y1, x2, y2]]).astype(np.float32)
        res['mpii_scale'] = s

        res['image'] = img
        res['path'] = img_path
        w, h = get_image_size(img)
        res['mpii_length'] = self.length

        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=img_path,
            point=np.array(a['joint_self'])[..., :2][np.newaxis, ...].astype(np.float32),
            point_meta=meta
        )

    def __len__(self):
        return len(self.data_lines)

    def __repr__(self):
        return 'MPIIReader(root={}, set_path={}, length={}, {})'.format(self.root, self.set_path, self.length, super(MPIIH5Reader, self).__repr__())
