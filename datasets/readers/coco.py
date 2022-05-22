import os
import numpy as np
from addict import Dict
from PIL import Image
from utils.bbox_tools import xywh2xyxy
from .reader import Reader
from .builder import READER


__all__ = ['COCOAPIReader', 'COCOBboxTxtReader', 'COCODNReader']


coco_classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
    'hair drier', 'toothbrush')


@READER.register_module()
class COCOAPIReader(Reader):
    def __init__(self, set_path, img_root, **kwargs):
        super(COCOAPIReader, self).__init__(**kwargs)

        self.set = set_path
        self.img_root = img_root
        self.classes = coco_classes
        self.use_instance_mask = False
        self.use_keypoint = False

        assert os.path.exists(self.set)
        assert os.path.exists(self.img_root)

        from pycocotools.coco import COCO

        self.coco_api = COCO(self.set)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        # self.cats = self.coco_api.loadCats(self.cat_ids)
        self.img_ids = sorted(self.coco_api.imgs.keys())
        self.data_info = self.coco_api.loadImgs(self.img_ids)

    def get_dataset_info(self):
        return range(len(self.img_ids)), Dict({'classes': self.classes, 'api': self.coco_api, 'ids': self.cat_ids})

    def get_data_info(self, index):
        img_info = self.data_info[index]
        anno = self.read_annotations(index)
        bbox = np.concatenate([anno['bboxes'], anno['labels'][..., np.newaxis]], axis=1)

        return dict(h=img_info['height'], w=img_info['width'], bbox=bbox)

    def read_annotations(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco_api.getAnnIds([img_id])
        anns = self.coco_api.loadAnns(ann_ids)
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []

        if self.use_instance_mask:
            gt_masks = []
        if self.use_keypoint:
            gt_keypoints = []

        for ann in anns:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                if self.use_instance_mask:
                    gt_masks.append(self.coco_api.annToMask(ann))
                if self.use_keypoint:
                    gt_keypoints.append(ann['keypoints'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        annotation = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        if self.use_instance_mask:
            annotation['masks'] = gt_masks

        if self.use_keypoint:
            if gt_keypoints:
                annotation['keypoints'] = np.array(gt_keypoints, dtype=np.float32)
            else:
                annotation['keypoints'] = np.zeros((0, 51), dtype=np.float32)

        return annotation

    def __call__(self, index):
        # index = data_dict

        img_info = self.data_info[index]
        # id_ = img_info['id']
        # if not isinstance(id_, int):
        #     raise TypeError('Image id must be int.')

        path = os.path.join(self.img_root, img_info['file_name'])
        # img = Image.open(path).convert('RGB')
        img = self.read_image(path)
        w, h = img_info['width'], img_info['height']

        anno = self.read_annotations(index)
        # print(path)
        # print(anno)
        # np.savetxt('a', anno['bboxes'])
        # exit()
        bbox = np.concatenate([anno['bboxes'], anno['labels'][..., np.newaxis]], axis=1)
        # difficult = np.zeros(bbox.shape[0])
        # return {'image': img, 'ori_size': np.array([h, w]).astype(np.float32), 'path': path, 'bbox': np.concatenate([bbox, np.full((len(bbox), 1), 1).astype(np.float32)], axis=1), 'bbox_ignore': anno['bboxes_ignore'], 'id': img_info['id']}
        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=path,
            bbox=np.concatenate([bbox, np.full((len(bbox), 1), 1).astype(np.float32)], axis=1),
            bbox_ignore=anno['bboxes_ignore'], 
            id=img_info['id']
        )

    def __repr__(self):
        return 'COCOAPIReader(set_path={}, img_root={}, classes={}, {})'.format(self.set, self.img_root, self.classes, super(COCOAPIReader, self).__repr__())


@READER.register_module()
class COCOBboxTxtReader(Reader):
    def __init__(self, txt_root, img_root, classes=coco_classes, **kwargs):
        super(COCOBboxTxtReader, self).__init__(**kwargs)

        self.txt_root = txt_root
        self.img_root = img_root
        self.classes = classes

        assert os.path.exists(self.txt_root)
        assert os.path.exists(self.img_root)

        self.img_paths = sorted(os.listdir(self.img_root))
        assert len(self.img_paths) > 0

    def get_dataset_info(self):
        return range(len(self.img_paths)), Dict({'classes': self.classes})

    def get_data_info(self, index):
        img = Image.open(self.image_paths[index])
        w, h = img.size
        bbox = self.read_bbox_coco(index, img)
        return dict(h=h, w=w, bbox=bbox)

    def read_annotations(self, name):
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []

        lines = [line.strip() for line in open(os.path.join(self.txt_root, name + '.txt'))]
        for line in lines:
            tmp = line.split('\t')
            tmp = [float(t) for t in tmp]
            x1, y1, x2, y2, c = tmp
            if c == -1:
                gt_bboxes_ignore.append([x1, y1, x2, y2])
            else:
                gt_bboxes.append([x1, y1, x2, y2])
                gt_labels.append(c)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
            # print(name, gt_bboxes_ignore)
            # exit()
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        annotation = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        return annotation

    def __call__(self, index):
        # index = data_dict

        name = os.path.splitext(self.img_paths[index])[0]
        path = os.path.join(self.img_root, self.img_paths[index])
        # img = Image.open(path).convert('RGB')
        img = self.read_image(path)
        w, h = img.size

        anno = self.read_annotations(name)
        # print(path)
        # print(anno)
        # exit()
        bbox = np.concatenate([anno['bboxes'], anno['labels'][..., np.newaxis]], axis=1)
        # return {'image': img, 'ori_size': np.array([h, w]).astype(np.float32), 'path': path, 'bbox': np.concatenate([bbox, np.full((len(bbox), 1), 1).astype(np.float32)], axis=1), 'bbox_ignore': anno['bboxes_ignore']}
        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=path,
            bbox=np.concatenate([bbox, np.full((len(bbox), 1), 1).astype(np.float32)], axis=1),
            bbox_ignore=anno['bboxes_ignore']
        )

    def __repr__(self):
        return 'COCOBboxTxtReader(txt_root={}, img_root={}, classes={}, {})'.format(self.txt_root, self.img_root, self.classes, super(COCOBboxTxtReader, self).__repr__())


@READER.register_module()
class COCODNReader(Reader):
    def __init__(self, set_path, classes=coco_classes, **kwargs):
        super(COCODNReader, self).__init__(**kwargs)

        self.set = set_path
        self.image_paths = [id_.rstrip() for id_ in open(self.set)]

        assert len(self.image_paths) > 0
        self.label_paths = ['.'.join(path.replace("images", "labels").split('.')[:-1] + ['txt']) for path in self.image_paths]

        self.classes = classes

    def get_dataset_info(self):
        return range(len(self.image_paths)), Dict({'classes': self.classes})

    def get_data_info(self, index):
        img = Image.open(self.image_paths[index])
        w, h = img.size
        bbox = self.read_bbox_coco(index, img)
        return dict(h=h, w=w, bbox=bbox)

    def read_bbox_coco(self, index, image):
        if not os.path.exists(self.label_paths[index]):
            return np.zeros((0, 5)).astype(np.float32)

        bboxes = np.loadtxt(self.label_paths[index]).reshape(-1, 5)
        w, h = image.size

        coor = bboxes[:, 1:]
        label = bboxes[:, :1]
        if not (label < len(self.classes)).all():
            raise ValueError('{}, {}'.format(self.label_paths[index], label))

        coor[:, 0] *= w
        coor[:, 1] *= h
        coor[:, 2] *= w
        coor[:, 3] *= h
        coor = xywh2xyxy(coor).astype(np.int32)

        bboxes = np.concatenate([coor, label], axis=1).astype(np.float32)
        return bboxes

    def __call__(self, index):
        # index = data_dict
        # img = Image.open(self.image_paths[index]).convert('RGB')
        img = self.read_image(self.image_paths[index])
        w, h = img.size
        bbox = self.read_bbox_coco(index, img)
        # difficult = np.zeros(bbox.shape[0])
        path = self.image_paths[index]
        # return {'image': img, 'ori_size': np.array([h, w]).astype(np.float32), 'path': path, 'bbox': np.concatenate([bbox, np.full((len(bbox), 1), 1).astype(np.float32)], axis=1)}
        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=path,
            bbox=np.concatenate([bbox, np.full((len(bbox), 1), 1).astype(np.float32)], axis=1)
        )

    def __repr__(self):
        return 'COCODNReader(set={}, classes={}, {})'.format(self.set, self.classes, super(COCODNReader, self).__repr__())
