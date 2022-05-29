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
from ..utils.warp_tools import get_image_size


__all__ = ['VOCReader', 'VOCSegReader', 'SBDReader', 'VOCCLSReader', 'VOCLikeSegReader']


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


@READER.register_module()
class VOCCLSReader(Reader):
    def __init__(self, root, obj_name, classes, split=None, separator='+=+', **kwargs):
        super(VOCCLSReader, self).__init__(**kwargs)

        img_root = os.path.join(root, 'JPEGImages')
        xml_root = os.path.join(root, 'Annotations')

        assert os.path.exists(img_root)
        assert os.path.exists(xml_root)

        self.root = root
        self.split = split
        self.separator = separator
        self.obj_name = obj_name

        self.classes = classes
        self.num_classes = 0

        for cg in classes:
            assert len(cg) > 0 and isinstance(cg, tuple)
            for c in cg:
                assert isinstance(c, str)
                self.num_classes += 1

        if split is not None:
            id_list_file = os.path.join(self.root, '{}.txt'.format(self.split))
            assert os.path.exists(id_list_file)
            image_paths = [id_.strip() for id_ in open(id_list_file)]
        else:
            image_paths = read_image_paths(img_root)

        assert len(image_paths) > 0
        label_paths = [os.path.join(xml_root, ntpath.basename(id_).split('.')[0] + '.xml') for id_ in image_paths]

        self.data_lines = []

        for ip, lp in zip(image_paths, label_paths):
            bboxes = self.read_bbox_voc(lp)
            for b in bboxes:
                self.data_lines.append([ip] + b)
        #     break
        # print(self.data_lines)

    def get_dataset_info(self):
        return range(len(self.data_lines)), Dict({'classes': self.classes})

    def get_data_info(self, index):
        img = Image.open(self.image_paths[index])
        w, h = img.size
        bbox, _ = self.read_bbox_voc(index)
        return dict(h=h, w=w, bbox=bbox)

    def read_bbox_voc(self, path):
        root = ET.parse(path).getroot()

        bboxes = []
        objects = root.findall('object')
        for obj in objects:
            bbox = obj.find('bndbox')
            class_name = obj.find('name').text.lower().strip()
            obj_name = class_name.split(self.separator)[0]
            if obj_name != self.obj_name.lower():
                continue
            xmin = int(float(bbox.find('xmin').text.strip()))
            xmax = int(float(bbox.find('xmax').text.strip()))
            ymin = int(float(bbox.find('ymin').text.strip()))
            ymax = int(float(bbox.find('ymax').text.strip()))
            if xmin == xmax or ymin == ymax:
                continue

            label_name = class_name.split('+=+')[-1]
            if self.num_classes != len(label_name):
                raise ValueError
            labels = np.zeros(len(self.classes)).astype(np.long)

            # print(label_name)

            start = 0
            for i, cg in enumerate(self.classes):
                # print(start, len(cg))
                # print(label_name[start: start + len(cg)])

                tmp = label_name[start: start + len(cg)]
                if len(tmp) == 1:
                    labels[i] = int(tmp)
                else:
                    tmp = [int(t) for t in tmp]
                    if sum(tmp) != 1:
                        raise ValueError
                    labels[i] = tmp.index(1)
                start += len(cg)
            # print(labels)
            # exit()
            bboxes.append([xmin, ymin, xmax, ymax, np.array(labels).astype(np.long)])

        return bboxes

    def __call__(self, index):
        image_path, x1, y1, x2, y2, labels = self.data_lines[index]
        img = self.read_image(image_path)
        w, h = img.size
        path = image_path
        bbox = np.array([[x1, y1, x2, y2]]).astype(np.float)

        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=path,
            bbox=bbox,
            label=labels
        )

    def __repr__(self):
        return 'VOCCLSReader(root={}, obj_name={}, classes={}, split={}, {})'.format(self.root, self.obj_name, self.classes, self.split, super(VOCCLSReader, self).__repr__())


@READER.register_module()
class VOCLikeSegReader(Reader):
    def __init__(self, root, cls_and_clr, split=None, **kwargs):
        super(VOCLikeSegReader, self).__init__(**kwargs)

        self.root = root

        self.classes = []
        self.colors = []
        if isinstance(cls_and_clr[0][1], tuple) or isinstance(cls_and_clr[0][1], list):
            self.nc_clr = len(cls_and_clr[0][1])
        else:
            self.nc_clr = 1

        for cls_, clr_ in cls_and_clr:
            if isinstance(clr_, tuple) or isinstance(clr_, list):
                nc_clr = len(clr_)
            else:
                nc_clr = 1

            assert nc_clr == self.nc_clr

            self.classes.append(cls_)
            self.colors.append(clr_)

        img_root = os.path.join(self.root, 'JPEGImages')
        mask_root = os.path.join(self.root, 'Annotations')

        assert os.path.exists(img_root)
        assert os.path.exists(mask_root)
        assert self.classes[0] == '__background__'

        if split is None:
            self.split = None
            self.image_paths = read_image_paths(img_root)
        else:
            self.split = split
            id_list_file = os.path.join(self.root, '{}.txt'.format(self.split))
            assert os.path.exists(id_list_file)

            self.image_paths = []
            folders = []
            for id_ in open(id_list_file):
                if len(id_.strip()) > 0:
                    folders.append(os.path.join(img_root, id_.strip()))
            for folder in folders:
                self.image_paths += read_image_paths(folder)

        assert len(self.image_paths) > 0
        self.mask_paths = [os.path.splitext(id_.replace('JPEGImages', 'Annotations'))[0] + '.png' for id_ in self.image_paths]

    def get_dataset_info(self):
        return range(len(self.image_paths)), Dict({'classes': self.classes})

    def get_data_info(self, index):
        img = Image.open(self.image_paths[index])
        w, h = img.size
        return dict(h=h, w=w)

    def __call__(self, index):
        img = self.read_image(self.image_paths[index])

        mask = Image.open(self.mask_paths[index])
        if self.nc_clr == 1:
            mask = mask.convert('L')
        else:
            mask = mask.convert('RGB')
        mask = np.array(mask)

        for i, color in enumerate(self.colors):
            mask[mask == color] = i

        w, h = img.size
        path = self.image_paths[index]

        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=path,
            mask=Image.fromarray(mask)
        )

    def __repr__(self):
        return 'VOCLikeSegReader(root={}, split={}, classes={}, {})'.format(self.root, self.split, self.classes, super(VOCLikeSegReader, self).__repr__())
