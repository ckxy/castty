import os
import pickle
import numpy as np
from .reader import Reader
from .builder import READER
from ..utils.structures import Meta
from ..utils.common import get_image_size


__all__ = ['DDI100SubsetDetReader']


@READER.register_module()
class DDI100SubsetDetReader(Reader):
    def __init__(self, root, **kwargs):
        super(DDI100SubsetDetReader, self).__init__(**kwargs)

        self.root = root

        self.img_root = os.path.join(self.root, 'orig_masks')
        self.pkl_root = os.path.join(self.root, 'orig_boxes')

        assert os.path.exists(self.img_root)
        assert os.path.exists(self.pkl_root)

        self.image_paths = sorted(os.listdir(self.img_root))
        self.pkl_paths = sorted(os.listdir(self.pkl_root))

        self._info = dict(
            forcat=dict(
                poly=dict(classes=['text'])
            ),
            tag_mapping=dict(
                image=['image'],
                poly=['poly'],
            )
        )

    def __getitem__(self, index):
        # index = 13
        # print(index)
        path = os.path.join(self.img_root, self.image_paths[index])

        img = self.read_image(path)
        w, h = get_image_size(img)

        with open(os.path.join(self.pkl_root, self.pkl_paths[index]), "rb") as f:
            data = pickle.load(f)

        polys = []
        keep_flags = []
        for d in data:
            polys.append(d['box'][[0, 1, 3, 2], ::-1].astype(np.float32))
            keep_flags.append(True)

        meta = Meta(
            class_id=np.zeros(len(polys)).astype(np.int32),
            keep=np.array(keep_flags),
        )

        return dict(
            image=img,
            image_meta=dict(ori_size=(w, h), path=path),
            poly=polys,
            poly_meta=meta
        )

    def __len__(self):
        return len(self.image_paths)

    def __repr__(self):
        return 'DDI100SubsetDetReader(root={}, {})'.format(self.root, super(DDI100SubsetDetReader, self).__repr__())
