import os
import json
import ntpath
import numpy as np
from addict import Dict
from PIL import Image
from .reader import Reader
from .utils import read_image_paths


__all__ = ['TextRendererReader', 'Auto7SegReader', 'CharSegMLReader']


class TextRendererReader(Reader):
    def __init__(self, root, chars_path, **kwargs):
        super(TextRendererReader, self).__init__(**kwargs)

        self.root = root
        self.chars_path = chars_path
        self.chars = [c.strip() for c in open(chars_path)] + ['-']

        data_lines = [c.strip() for c in open(os.path.join(root, 'tmp_labels.txt'))]

        self.img_paths = []
        self.seqs = []
        for data_line in data_lines:
            s = data_line.split(' ')
            self.img_paths.append(s[0] + '.jpg')
            self.seqs.append(' '.join(s[1:]))
            # print(data_line.split(' '))
        # print(self.img_paths)
        # print(self.seqs)
        # exit()
        assert len(self.img_paths) > 0

    def get_dataset_info(self):
        return range(len(self.img_paths)), Dict({'chars': self.chars})

    def get_data_info(self, index):
        img = Image.open(self.img_paths[index][0])
        w, h = img.size
        return dict(h=h, w=w)

    def __call__(self, index):
        # index = data_dict
        # img = Image.open(os.path.join(self.root, self.img_paths[index])).convert('RGB')
        img = self.read_image(os.path.join(self.root, self.img_paths[index]))
        w, h = img.size
        path = os.path.join(self.root, self.img_paths[index])

        words = []
        for c in self.seqs[index]:
            words.append(self.chars.index(c))

        # return {'image': img, 'ori_size': np.array([h, w]).astype(np.float32), 'path': path, 'seq': words, 'seq_length': len(words)}
        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=path,
            seq=words,
            seq_length=len(words)
        )

    def __repr__(self):
        return 'TextRendererReader(root={}, chars_path={}, {})'.format(self.root, self.chars_path, super(TextRendererReader, self).__repr__())


class Auto7SegReader(Reader):
    def __init__(self, root, **kwargs):
        super(Auto7SegReader, self).__init__(**kwargs)

        self.root = root
        self.chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        self.img_paths = read_image_paths(root)

        assert len(self.img_paths) > 0

    def get_dataset_info(self):
        return range(len(self.img_paths)), Dict({'chars': self.chars})

    def get_data_info(self, index):
        img = Image.open(self.img_paths[index][0])
        w, h = img.size
        return dict(h=h, w=w)

    def __call__(self, index):
        # index = data_dict
        # img = Image.open(os.path.join(self.root, self.img_paths[index])).convert('RGB')
        img = self.read_image(os.path.join(self.root, self.img_paths[index]))
        w, h = img.size
        path = os.path.join(self.root, self.img_paths[index])

        word = os.path.splitext(ntpath.basename(path))[0]
        word = word.split('_')[-1]

        # return {'image': img, 'ori_size': np.array([h, w]).astype(np.float32), 'path': path, 'seq': word}
        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=path,
            seq=words
        )

    def __repr__(self):
        return 'Auto7SegReader(root={}, {})'.format(self.root, super(Auto7SegReader, self).__repr__())


class CharSegMLReader(Reader):
    def __init__(self, root, **kwargs):
        super(CharSegMLReader, self).__init__(**kwargs)

        self.root = root
        self.word_bboxes = []
        self.char_polygons = []
        self.paths = []
        self.lines = []

        paths = read_image_paths(root)
        self.paths += paths[1:]

        with open(os.path.join(self.root, '0.json'), 'r') as f:
            load_dict = json.load(f)

        w, h = load_dict['imageWidth'], load_dict['imageHeight']

        polygons = [[] for _ in range(8)]
        for i, s in enumerate(load_dict['shapes']):
            if s['shape_type'] != 'polygon':
                continue
            if s['label'] == '0':
                continue
            pts = np.array(s['points']).astype(np.int)
            ind = int(s['label']) - 1
            polygons[ind].append(pts[np.newaxis, ...])

        for i in range(8):
            polygons[i] = np.concatenate(polygons[i])

        self.word_bboxes = []
        for i in range(8):                
            tmp = polygons[i].reshape(-1, 2)
            xmin = np.min(tmp[:, 0])
            xmax = np.max(tmp[:, 0])
            ymin = np.min(tmp[:, 1])
            ymax = np.max(tmp[:, 1])

            polygons[i][..., 0] -= xmin
            polygons[i][..., 1] -= ymin
            self.word_bboxes.append((xmin, ymin, xmax, ymax))

        self.char_polygons = polygons

        self.lines = [id_.strip() for id_ in open(os.path.join(self.root, '0.txt'))]

        assert len(self.paths) > 0

    def get_dataset_info(self):
        return range(len(self.paths) * 8), Dict({})

    def get_data_info(self, index):
        word_id = index % 8
        path = self.paths[img_id]
        img = Image.open(path).convert('RGB')
        w, h = img.size
        return dict(h=h, w=w)

    def __call__(self, index):
        # index = data_dict
        img_id = index // 8
        word_id = index % 8
        path = self.paths[img_id]

        # image = Image.open(path).convert('RGB')
        image = self.read_image(path)
        polygons = self.char_polygons[word_id].astype(np.float32)
        line = self.lines[img_id]
        word = line.split('\t')[word_id + 1]

        word_bbox = self.word_bboxes[word_id]

        image = image.crop(word_bbox)
        w, h = image.size

        # return {'image': image, 'ori_size': np.array([h, w]).astype(np.float32), 'path': path, 'seq': word, 'polygon': polygons}
        return dict(
            image=image,
            ori_size=np.array([h, w]).astype(np.float32),
            path=path,
            seq=word,
            quad=polygons
        )

    def __repr__(self):
        return 'CharSegMLReader(root={}, {})'.format(self.root, super(CharSegMLReader, self).__repr__())
