import os
import numpy as np
from .reader import Reader
from .builder import READER
from .utils import is_image_file
from ..utils.structures import Meta
from ..utils.common import get_image_size


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

        self._info = dict(
            forcat=dict(
                label=dict(
                    classes=self.classes
                ),
            ),
            tag_mapping=dict(
                image=['image'],
                label=['label']
            )
        )

    def __getitem__(self, index):
        img = self.read_image(self.samples[index][0])
        w, h = get_image_size(img)
        path = self.samples[index][0]

        label = np.zeros(len(self.classes)).astype(np.float32)
        label[self.samples[index][1]] = 1

        return dict(
            image=img,
            image_meta=dict(ori_size=(w, h), path=path),
            label=[label],
        )

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return 'ImageFolderReader(root={}, classes={}, {})'.format(self.root, tuple(self.classes), super(ImageFolderReader, self).__repr__())
