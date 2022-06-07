import os
import ntpath
import numpy as np
from PIL import Image
from addict import Dict
from scipy.io import loadmat
from .reader import Reader
from .utils import read_image_paths
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from .builder import READER
from ..utils.structures import Meta
from ..utils.common import get_image_size


__all__ = ['VOCReader', 'VOCSegReader', 'SBDReader']


@READER.register_module()
class VOCReader(Reader):
    def __init__(self, root, classes, split=None, filter_difficult=False, to_remove=False, **kwargs):
        super(VOCReader, self).__init__(**kwargs)

        img_root = os.path.join(root, 'JPEGImages')
        xml_root = os.path.join(root, 'Annotations')

        assert os.path.exists(img_root)
        assert os.path.exists(xml_root)

        self.root = root
        self.classes = classes
        self.split = split

        if split is not None:
            id_list_file = os.path.join(self.root, 'ImageSets/Main/{}.txt'.format(self.split))
            assert os.path.exists(id_list_file)
            self.image_paths = [os.path.join(img_root, id_.strip() + '.jpg') for id_ in open(id_list_file)]
        else:
            self.image_paths = read_image_paths(img_root)

        assert len(self.image_paths) > 0
        self.label_paths = [os.path.join(xml_root, ntpath.basename(id_).split('.')[0] + '.xml') for id_ in self.image_paths]

        self.filter_difficult = filter_difficult

        if to_remove:
            self.to_remove = 1
        else:
            self.to_remove = 0

    def get_dataset_info(self):
        return range(len(self.image_paths)), Dict({'classes': self.classes})

    def get_data_info(self, index):
        img = Image.open(self.image_paths[index])
        w, h = img.size
        bbox, _ = self.read_bbox_voc(index)
        return dict(h=h, w=w, bbox=bbox)

    @staticmethod
    def read_bbox_voc(xml_path, classes, filter_difficult=False, to_remove=0):
        root = ET.parse(xml_path).getroot()

        bboxes = []
        classes_list = []
        difficult = []
        objects = root.findall('object')
        for obj in objects:
            if obj.find('difficult') is None:
                diff = 0
            else:
                diff = int(obj.find('difficult').text.strip())
            if filter_difficult and diff == 1:
                continue
            bbox = obj.find('bndbox')
            class_name = obj.find('name').text.lower().strip()
            if class_name not in classes:
                continue
            class_index = classes.index(class_name)
            xmin = int(float(bbox.find('xmin').text.strip())) - to_remove
            xmax = int(float(bbox.find('xmax').text.strip())) - to_remove
            ymin = int(float(bbox.find('ymin').text.strip())) - to_remove
            ymax = int(float(bbox.find('ymax').text.strip())) - to_remove

            if xmin == xmax or ymin == ymax:
                continue
            bboxes.append([xmin, ymin, xmax, ymax])
            classes_list.append(class_index)
            difficult.append(diff)

        if len(bbox) == 0:
            bboxes = np.zeros((0, 4)).astype(np.float32)
            classes_list = np.array(0)
            difficult = np.zeros(0)
        else:
            bboxes = np.array(bboxes).astype(np.float32)
            classes_list = np.array(classes_list)
            difficult = np.array(difficult)

        return bboxes, classes_list, difficult

    def __call__(self, index):
        img = self.read_image(self.image_paths[index])
        # w, h = img.size
        w, h = get_image_size(img)
        bbox, cla, difficult = self.read_bbox_voc(self.label_paths[index], self.classes, self.filter_difficult, self.to_remove)
        path = self.image_paths[index]
        bbox_meta = Meta(['class_id', 'score', 'difficult'], [cla, np.ones(len(bbox)), difficult])

        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=path,
            bbox=bbox,
            bbox_meta=bbox_meta
        )

    def __repr__(self):
        return 'VOCReader(root={}, split={}, classes={}, filter_difficult={}, to_remove={}, {})'.format(self.root, self.split, self.classes, self.filter_difficult, self.to_remove, super(VOCReader, self).__repr__())


@READER.register_module()
class VOCSegReader(Reader):
    def __init__(self, root, classes, split=None, **kwargs):
        super(VOCSegReader, self).__init__(**kwargs)

        self.root = root
        self.classes = classes

        img_root = os.path.join(self.root, 'JPEGImages')
        mask_root = os.path.join(self.root, 'SegmentationClass')

        assert os.path.exists(img_root)
        assert os.path.exists(mask_root)
        assert classes[0] == '__background__'

        if split is None:
            self.split = None
            self.image_paths = read_image_paths(img_root)
        else:
            self.split = split
            id_list_file = os.path.join(self.root, 'ImageSets/Segmentation/{}.txt'.format(self.split))
            assert os.path.exists(id_list_file)
            self.image_paths = [os.path.join(img_root, id_.strip() + '.jpg') for id_ in open(id_list_file)]

        assert len(self.image_paths) > 0
        self.mask_paths = [os.path.join(mask_root, ntpath.basename(id_).split('.')[0] + '.png') for id_ in self.image_paths]

    def get_dataset_info(self):
        return range(len(self.image_paths)), Dict({'classes': self.classes})

    def get_data_info(self, index):
        img = Image.open(self.image_paths[index])
        w, h = img.size
        return dict(h=h, w=w)

    def __call__(self, index):
        # index = data_dict
        # img = Image.open(self.image_paths[index]).convert('RGB')
        img = self.read_image(self.image_paths[index])
        mask = Image.open(self.mask_paths[index])
        w, h = img.size
        path = self.image_paths[index]
        # return {'image': img, 'ori_size': np.array([h, w]).astype(np.float32), 'path': path, 'mask': mask}
        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=path,
            mask=mask
        )

    def __repr__(self):
        return 'VOCSegReader(root={}, split={}, classes={}, {})'.format(self.root, self.split, self.classes, super(VOCSegReader, self).__repr__())


@READER.register_module()
class SBDReader(Reader):
    def __init__(self, root, classes, split, **kwargs):
        super(SBDReader, self).__init__(**kwargs)

        self.root = root
        self.classes = classes
        self.split = split

        img_root = os.path.join(self.root, 'img')
        mask_root = os.path.join(self.root, 'cls')

        assert os.path.exists(img_root)
        assert os.path.exists(mask_root)
        assert classes[0] == '__background__'

        id_list_file = os.path.join(self.root, '{}.txt'.format(self.split))

        self.image_paths = [os.path.join(img_root, id_.strip() + '.jpg') for id_ in open(id_list_file)]
        self.mask_paths = [os.path.join(mask_root, id_.strip() + '.mat') for id_ in open(id_list_file)]

        assert len(self.image_paths) > 0

    def get_dataset_info(self):
        return range(len(self.image_paths)), Dict({'classes': self.classes})

    def get_data_info(self, index):
        img = Image.open(self.image_paths[index])
        w, h = img.size
        return dict(h=h, w=w)

    def __call__(self, index):
        # index = data_dict
        # img = Image.open(self.image_paths[index]).convert('RGB')
        img = self.read_image(self.image_paths[index])
        mat = loadmat(self.mask_paths[index])
        mask = mat['GTcls'][0]['Segmentation'][0].astype(np.uint8)
        mask = Image.fromarray(mask, mode='P')
        w, h = img.size
        path = self.image_paths[index]
        # return {'image': img, 'ori_size': np.array([h, w]).astype(np.float32), 'path': path, 'mask': mask}
        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=path,
            mask=mask
        )

    def __repr__(self):
        return 'SBDReader(root={}, split={}, classes={}, {})'.format(self.root, self.split, self.classes, super(SBDReader, self).__repr__())