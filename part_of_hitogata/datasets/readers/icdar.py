import os
import numpy as np
from .reader import Reader
from .builder import READER
from ..utils.structures import Meta
from ..utils.common import get_image_size


__all__ = ['ICDARDetReader']


@READER.register_module()
class ICDARDetReader(Reader):
    def __init__(self, root, train=True, filter_texts=['###'], **kwargs):
        super(ICDARDetReader, self).__init__(**kwargs)

        self.root = root
        self.filter_texts = set(filter_texts)
        self.train = train
        if train:
            self.img_root = os.path.join(self.root, 'ch4_training_images')
            self.txt_root = os.path.join(self.root, 'ch4_training_localization_transcription_gt')
        else:
            self.img_root = os.path.join(self.root, 'ch4_test_images')
            self.txt_root = os.path.join(self.root, 'Challenge4_Test_Task1_GT')

        assert os.path.exists(self.img_root)
        assert os.path.exists(self.txt_root)

        self.image_paths = sorted(os.listdir(self.img_root))
        self.txt_paths = sorted(os.listdir(self.txt_root))

        self._info = dict(
            forcat=dict(
                type='ocrdet',
            )
        )

    def __call__(self, index):
        path = os.path.join(self.img_root, self.image_paths[index])

        img = self.read_image(path)
        w, h = get_image_size(img)

        lines = [id_.strip() for id_ in open(os.path.join(self.txt_root, self.txt_paths[index]), encoding='UTF-8-sig')]
        
        polys = []
        ignore_flags = []
        for l in lines:
            l = l.split(',')[:9]
            coords = l[:-1]
            if self.filter_texts and l[-1] in self.filter_texts:
                ignore_flags.append(True)
            else:
                ignore_flags.append(False)
            assert len(coords) == 8
            coords = [float(c) for c in coords]
            coords = np.array(coords).reshape(-1, 2)
            polys.append(coords)

        # polys.append(np.array([[460, 155], [510, 160], [515, 175], [470, 170]]).astype(np.float32))
        # ignore_flags.append(False)

        meta = Meta(ignore_flag=np.array(ignore_flags))

        # meta.append(['class_id'], [np.array([1, 1, 0, 0, 0, 0, 0], dtype=np.int32)])

        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=path,
            poly=polys,
            poly_meta=meta
        )

    def __len__(self):
        return len(self.image_paths)

    def __repr__(self):
        return 'ICDARDetReader(root={}, train={}, filter_texts={}, {})'.format(self.root, self.train, list(self.filter_texts), super(ICDARDetReader, self).__repr__())
