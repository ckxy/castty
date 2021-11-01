import os
import json
import numpy as np
from addict import Dict
from copy import deepcopy
from PIL import Image
from .reader import Reader


__all__ = ['LSPReader']


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

    def get_dataset_info(self):
        return range(len(self.data_lines)), Dict({})

    def get_data_info(self, index):
        return

    def __call__(self, index):
        # index = data_dict

        a = self.data_lines[index]
        img_path = os.path.join(self.root, a['img_paths'])
        # img = Image.open(img_path).convert('RGB')
        img = self.read_image(img_path)

        res = dict()
        res['image'] = img
        res['point'] = np.array(a['joint_self'])[..., :2].astype(np.float32)
        res['visible'] = np.array(a['joint_self'])[..., 2].astype(np.int)
        res['path'] = img_path
        w, h = img.size
        res['ori_size'] = np.array([h, w]).astype(np.float32)
        return res

    def __repr__(self):
        return 'LSPReader(root={}, set_path={}, is_test={}, {})'.format(self.root, self.set_path, self.is_test, super(LSPReader, self).__repr__())
