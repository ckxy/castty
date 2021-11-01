import os
import json
import ntpath
import numpy as np
from addict import Dict
from PIL import Image, ImageDraw
from .reader import Reader


__all__ = ['MEJSONReader', 'NewMEJSONReader', 'LabelmeMaskReader']


class MEJSONReader(Reader):# todo
    def __init__(self, root, txt_path, mode='6s', **kwargs):
        super(MEJSONReader, self).__init__(**kwargs)

        assert os.path.exists(root)
        assert os.path.exists(txt_path)
        assert mode in ['6s', '6s1p']
        self.root = root
        self.txt_path = txt_path
        self.mode = mode

        with open(self.txt_path, 'r') as f:
            self.data_lines = f.readlines()

        assert len(self.data_lines) > 0

    def get_dataset_info(self):
        return range(len(self.data_lines)), Dict({})

    def get_data_info(self, index):
        return

    def __call__(self, index):
        # index = data_dict

        img_path = os.path.join(self.root, self.data_lines[index].strip())
        point_path = os.path.join(self.root, os.path.splitext(self.data_lines[index].strip())[0].replace('img/', 'point/') + '.json')
        
        with open(point_path, 'r') as f:
            load_dict = json.load(f)

        box = np.zeros((0, 4))
        landmark = []
        for i, s in enumerate(load_dict['shapes']):
            if self.mode == '6s1p':
                if i == 0:
                    box = np.array(s['points']).flatten()
                else:
                    landmark.append(s['points'][0])
            elif self.mode == '6s':
                if i == 0:
                    box = np.array(s['points']).flatten()
                elif i < 7:
                    landmark.append(s['points'][0])
            else:
                raise ValueError('{}不是期望的数据读取模式'.format(self.mode))
        landmark = np.array(landmark).astype(np.float32)

        img = self.read_image(img_path)

        res = dict()
        res['image'] = img
        res['bbox'] = box[np.newaxis, ...].astype(np.float32)
        res['point'] = landmark
        res['path'] = img_path
        w, h = img.size
        res['ori_size'] = np.array([h, w]).astype(np.float32)
        return res

    def __repr__(self):
        return 'MEJSONReader(mode={}, root={}, txt_path={}, {})'.format(self.mode, self.root, self.txt_path, super(MEJSONReader, self).__repr__())


class NewMEJSONReader(Reader):
    def __init__(self, root, split=None, **kwargs):
        super(NewMEJSONReader, self).__init__(**kwargs)

        assert os.path.exists(root)
        self.root = root
        self.split = split

        assert os.path.exists(os.path.join(root, 'json'))
        assert os.path.exists(os.path.join(root, 'img'))

        if split is None:
            self.json_paths = sorted(os.listdir(os.path.join(root, 'json')))
        else:
            assert os.path.isfile(os.path.join(root, split + '.txt'))
            self.json_paths = [id_.strip() for id_ in open(os.path.join(root, split + '.txt'))]

        assert len(self.json_paths) > 0

    def get_dataset_info(self):
        return range(len(self.json_paths)), Dict({})

    def get_data_info(self, index):
        return

    def __call__(self, index):        
        with open(os.path.join(self.root, 'json', self.json_paths[index]), 'r') as f:
            load_dict = json.load(f)

        img_path = os.path.join(self.root, 'img', load_dict['imagePath'])

        box = np.zeros((0, 4))
        landmark = []
        for i, s in enumerate(load_dict['shapes']):
            if i == 0:
                box = np.array(s['points']).flatten()
            else:
                landmark.append(s['points'][0])

        landmark = np.array(landmark).astype(np.float32)

        img = self.read_image(img_path)

        res = dict()
        res['image'] = img
        res['bbox'] = box[np.newaxis, ...].astype(np.float32)
        res['point'] = landmark
        res['path'] = img_path
        w, h = img.size
        res['ori_size'] = np.array([h, w]).astype(np.float32)
        return res

    def __repr__(self):
        return 'NewMEJSONReader(root={}, split={}, {})'.format(self.root, self.split, super(NewMEJSONReader, self).__repr__())


class LabelmeMaskReader(Reader):
    def __init__(self, root, classes, **kwargs):
        super(LabelmeMaskReader, self).__init__(**kwargs)

        self.root = root
        self.classes = classes

        self.img_root = os.path.join(self.root, 'img')
        mask_root = os.path.join(self.root, 'json')

        assert os.path.exists(self.img_root)
        assert os.path.exists(mask_root)
        assert self.classes[0] == '__background__'

        self.mask_paths = sorted(os.listdir(mask_root))
        self.mask_paths = [os.path.join(mask_root, path) for path in self.mask_paths]

        assert len(self.mask_paths) > 0

    def get_dataset_info(self):
        return range(len(self.mask_paths)), Dict({'classes': self.classes})

    def get_data_info(self, index):
        with open(self.mask_paths[index], 'r') as f:
            load_dict = json.load(f)
        w, h = load_dict['imageWidth'], load_dict['imageHeight']
        return dict(h=h, w=w)

    def __call__(self, index):
        # index = data_dict

        with open(self.mask_paths[index], 'r') as f:
            load_dict = json.load(f)

        w, h = load_dict['imageWidth'], load_dict['imageHeight']
        mask = Image.new('P', (w, h), 0)

        for s in load_dict['shapes']:
            if s['shape_type'] != 'polygon':
                continue
            if s['label'] not in self.classes:
                continue
            # print(pts)
            # print(s['label'], self.classes.index(s['label']))
            pts = np.array(s['points']).astype(np.int).flatten().tolist()
            ImageDraw.Draw(mask).polygon(pts, fill=self.classes.index(s['label']))

        path = os.path.join(self.img_root, ntpath.basename(load_dict['imagePath']))

        # img = Image.open(path).convert('RGB')
        img = self.read_image(path)
        # return {'image': img, 'ori_size': np.array([h, w]).astype(np.float32), 'path': path, 'mask': mask}
        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=path,
            mask=mask
        )

    def __repr__(self):
        return 'LabelmeMaskReader(root={}, classes={}, {})'.format(self.root, self.classes, super(LabelmeMaskReader, self).__repr__())
