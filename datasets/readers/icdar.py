import os
import numpy as np
from addict import Dict
from .reader import Reader
from .builder import READER
from ..utils.common import get_image_size


__all__ = ['ICDARDetReader']


@READER.register_module()
class ICDARDetReader(Reader):
    def __init__(self, root, **kwargs):
        super(ICDARDetReader, self).__init__(**kwargs)

        self.root = root
        self.img_root = os.path.join(self.root, 'ch4_training_images')
        self.txt_root = os.path.join(self.root, 'ch4_training_localization_transcription_gt')

        assert os.path.exists(self.img_root)
        assert os.path.exists(self.txt_root)

        self.image_paths = sorted(os.listdir(self.img_root))
        self.txt_paths = sorted(os.listdir(self.txt_root))

    def get_dataset_info(self):
        return range(len(self.image_paths)), Dict({})

    def get_data_info(self, index):
        pass

    def __call__(self, index):
        path = os.path.join(self.img_root, self.image_paths[index])

        img = self.read_image(path)
        w, h = get_image_size(img)

        lines = [id_.strip() for id_ in open(os.path.join(self.txt_root, self.txt_paths[index]), encoding='UTF-8-sig')]
        
        poly = []
        for l in lines:
            l = l.split(',')
            coords = l[:-1]
            assert len(coords) == 8
            coords = [float(c) for c in coords]
            coords = np.array(coords).reshape(-1, 2)
            poly.append(coords)     

        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=path,
            poly=poly
        )

    def __repr__(self):
        return 'ICDARDetReader(root={}, {})'.format(self.root, super(ICDARDetReader, self).__repr__())
