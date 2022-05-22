import os
import numpy as np
from addict import Dict
from PIL import Image
from .reader import Reader
from .builder import READER


__all__ = ['WFLWReader', 'WFLWPPReader']


@READER.register_module()
class WFLWReader(Reader):
    def __init__(self, root, txt_path, **kwargs):
        super(WFLWReader, self).__init__(**kwargs)

        self.root = root
        self.txt = txt_path
        self.img_root = os.path.join(self.root, 'WFLW_images')
        assert os.path.exists(self.img_root)

        with open(os.path.join(self.root, self.txt), 'r') as f:
            self.data_lines = f.readlines()

        assert len(self.data_lines) > 0

    def get_dataset_info(self):
        return range(len(self.data_lines)), Dict({})

    def get_data_info(self, index):
        return

    def __call__(self, index):
        # index = data_dict

        line = self.data_lines[index].strip().split()
        landmark = np.asarray(list(map(float, line[:196])), dtype=np.float32).reshape(-1, 2)
        box = np.asarray(list(map(int, line[196:200])))
        attribute = np.asarray(list(map(int, line[200:206])), dtype=np.int)
        name = line[206]

        # img = Image.open(os.path.join(self.img_root, name)).convert('RGB')
        img = self.read_image(os.path.join(self.img_root, name))

        res = dict()
        res['image'] = img
        res['point'] = landmark
        res['path'] = os.path.join(self.img_root, name)
        res['attribute'] = attribute
        res['bbox'] = box[np.newaxis, ...].astype(np.float)
        w, h = img.size
        res['ori_size'] = np.array([h, w]).astype(np.float32)
        return res

    def __repr__(self):
        return 'WFLWReader(txt={}, {})'.format(self.txt, super(WFLWReader, self).__repr__())


@READER.register_module()
class WFLWPPReader(Reader):
    def __init__(self, txt_path, **kwargs):
        super(WFLWPPReader, self).__init__(**kwargs)

        self.txt = txt_path
        with open(txt_path, 'r') as f:
            self.data_lines = f.readlines()
        assert len(self.data_lines) > 0

    def get_dataset_info(self):
        return range(len(self.data_lines)), Dict({})

    def get_data_info(self, index):
        return

    def __call__(self, index):
        # index = data_dict

        line = self.data_lines[index].strip().split()
        path = line[0]
        landmark = np.asarray(line[1:197], dtype=np.float32).reshape(-1, 2)
        # landmark = landmark.reshape(-1, 2)
        attribute = np.asarray(line[197:203], dtype=np.int32)
        euler_angle = np.asarray(line[203:206], dtype=np.float32)

        # img = Image.open(path).convert('RGB')
        img = self.read_image(path)

        res = dict()
        res['image'] = img
        res['point'] = landmark
        res['path'] = path
        res['attribute'] = attribute
        res['euler_angle'] = euler_angle
        w, h = img.size
        res['ori_size'] = np.array([h, w]).astype(np.float32)
        return res

    def __repr__(self):
        return 'WFLWPPReader(txt={}, {})'.format(self.txt, super(WFLWPPReader, self).__repr__())
