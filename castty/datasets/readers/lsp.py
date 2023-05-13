import os
import json
import numpy as np
from .reader import Reader
from .builder import READER
from ..utils.structures import Meta
from ..utils.common import get_image_size


__all__ = ['LSPReader']


@READER.register_module()
class LSPReader(Reader):
    def __init__(self, root, set_path, is_test=False, **kwargs):
        super(LSPReader, self).__init__(**kwargs)

        assert os.path.exists(root)
        assert os.path.exists(set_path)

        self.root = root
        self.set_path = set_path
        self.is_test = is_test

        with open(self.set_path) as anno_file:
            anno = json.load(anno_file)

        self.data_lines = []
        for idx, val in enumerate(anno):
            if val['isValidation'] == self.is_test:
                self.data_lines.append(val)

        assert len(self.data_lines) > 0

        self._info = dict(
            forcat=dict(
                point=dict(
                    classes=[str(i) for i in range(16)],
                )
            ),
            tag_mapping=dict(
                image=['image'],
                point=['point']
            )
        )

    def __getitem__(self, index):
        data_line = self.data_lines[index]

        img_path = os.path.join(self.root, data_line['img_paths'])
        img = self.read_image(img_path)
        w, h = get_image_size(img)

        meta = Meta(keep=np.array(data_line['joint_self'])[..., 2][np.newaxis, ...].astype(np.bool_))

        return dict(
            image=img,
            image_meta=dict(ori_size=(w, h), path=img_path),
            point=np.array(data_line['joint_self'])[..., :2][np.newaxis, ...].astype(np.float32),
            point_meta=meta
        )

    def __len__(self):
        return len(self.data_lines)

    def __repr__(self):
        return 'LSPReader(root={}, set_path={}, is_test={}, {})'.format(self.root, self.set_path, self.is_test, super(LSPReader, self).__repr__())
