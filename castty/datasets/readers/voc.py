import os
import numpy as np
from PIL import Image
from .reader import Reader
from .builder import READER
from scipy.io import loadmat
from ..utils.structures import Meta
from .utils import read_image_paths
from ..utils.common import get_image_size
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


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
        self.label_paths = [os.path.join(xml_root, os.path.basename(id_).split('.')[0] + '.xml') for id_ in self.image_paths]

        self.filter_difficult = filter_difficult

        if to_remove:
            self.to_remove = 1
        else:
            self.to_remove = 0

        self._info = dict(
            forcat=dict(
                bbox=dict(
                    classes=self.classes,
                    extra_meta=['difficult']
                )
            ), 
            tag_mapping=dict(
                image=['image'],
                bbox=['bbox']
            )
        )

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

    def __getitem__(self, index):
        img = self.read_image(self.image_paths[index])
        w, h = get_image_size(img)
        bbox, cla, difficult = self.read_bbox_voc(self.label_paths[index], self.classes, self.filter_difficult, self.to_remove)
        path = self.image_paths[index]

        bbox_meta = Meta(
            class_id=cla,
            score=np.ones(len(bbox)).astype(np.float32),
            difficult=difficult,
            keep=np.ones(len(bbox)).astype(np.bool_),
        )

        return dict(
            image=img,
            image_meta=dict(ori_size=(w, h), path=path),
            bbox=bbox,
            bbox_meta=bbox_meta
        )

    def __len__(self):
        return len(self.image_paths)

    def __repr__(self):
        return 'VOCReader(root={}, split={}, classes={}, filter_difficult={}, to_remove={}, {})'.format(self.root, self.split, self.classes, self.filter_difficult, self.to_remove, super(VOCReader, self).__repr__())


@READER.register_module()
class VOCSegReader(Reader):
    def __init__(self, root, classes, split=None, ignore_contour=True, **kwargs):
        super(VOCSegReader, self).__init__(**kwargs)

        assert classes[0] == '__background__'

        self.root = root
        self.ignore_contour = ignore_contour

        if not ignore_contour:
            self.classes = list(classes) + ['contour']
            self.classes = tuple(self.classes)
        else:
            self.classes = tuple(classes)

        img_root = os.path.join(self.root, 'JPEGImages')
        mask_root = os.path.join(self.root, 'SegmentationClass')

        assert os.path.exists(img_root)
        assert os.path.exists(mask_root)

        if split is None:
            self.split = None
            self.image_paths = read_image_paths(img_root)
        else:
            self.split = split
            id_list_file = os.path.join(self.root, 'ImageSets/Segmentation/{}.txt'.format(self.split))
            assert os.path.exists(id_list_file)
            self.image_paths = [os.path.join(img_root, id_.strip() + '.jpg') for id_ in open(id_list_file)]

        assert len(self.image_paths) > 0
        self.mask_paths = [os.path.join(mask_root, os.path.basename(id_).split('.')[0] + '.png') for id_ in self.image_paths]

        self._info = dict(
            forcat=dict(
                mask=dict(
                    classes=self.classes
                )
            ),
            tag_mapping=dict(
                image=['image'],
                mask=['mask']
            )
        )

    def __getitem__(self, index):
        img = self.read_image(self.image_paths[index])
        w, h = get_image_size(img)

        mask = Image.open(self.mask_paths[index])
        mask = np.array(mask).astype(np.int32)

        if self.ignore_contour:
            mask[mask == 255] = 0
        else:
            mask[mask == 255] = len(self.classes) - 1

        return dict(
            image=img,
            image_meta=dict(ori_size=(w, h), path=self.image_paths[index]),
            mask=mask
        )

    def __len__(self):
        return len(self.image_paths)

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

        self._info = dict(
            forcat=dict(
                mask=dict(
                    classes=self.classes
                )
            ),
            tag_mapping=dict(
                image=['image'],
                mask=['mask']
            )
        )

    def __getitem__(self, index):
        img = self.read_image(self.image_paths[index])
        w, h = get_image_size(img)

        mat = loadmat(self.mask_paths[index])
        mask = mat['GTcls'][0]['Segmentation'][0].astype(np.int32)

        return dict(
            image=img,
            image_meta=dict(ori_size=(w, h), path=self.image_paths[index]),
            mask=mask
        )

    def __len__(self):
        return len(self.image_paths)

    def __repr__(self):
        return 'SBDReader(root={}, split={}, classes={}, {})'.format(self.root, self.split, self.classes, super(SBDReader, self).__repr__())
