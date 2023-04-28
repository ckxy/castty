import os
import numpy as np
from PIL import Image
from .reader import Reader
from .builder import READER
from ..utils.structures import Meta
from ...utils.bbox_tools import xywh2xyxy


__all__ = ['COCOAPIReader']


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
    def __init__(self, set_path, img_root, classes=coco_classes, use_keypoint=False, **kwargs):
        super(COCOAPIReader, self).__init__(**kwargs)

        self.set = set_path
        self.img_root = img_root
        self.classes = classes
        self.use_instance_mask = False
        self.use_keypoint = use_keypoint

        if self.use_keypoint:
            self.classes = ['person']

        assert os.path.exists(self.set)
        assert os.path.exists(self.img_root)

        from pycocotools.coco import COCO

        self.coco_api = COCO(self.set)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        # self.cats = self.coco_api.loadCats(self.cat_ids)
        self.img_ids = sorted(self.coco_api.imgs.keys())
        self.data_info = self.coco_api.loadImgs(self.img_ids)

        self._info = dict(
            forcat=dict(
                bbox=dict(
                    classes=self.classes
                ),
            ),
            api=self.coco_api,
            ids=self.cat_ids
        )

        if self.use_keypoint:
            self._info['forcat']['point'] = dict()
            # self._info['forcat']['point']

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
                    gt_keypoints.append(np.array(ann['keypoints'], dtype=np.float32).reshape(1, -1, 3))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int32)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int32)

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
                annotation['keypoints'] = np.concatenate(gt_keypoints, axis=0)
            else:
                annotation['keypoints'] = np.zeros((0, 17, 3), dtype=np.float32)

        return annotation

    def __getitem__(self, index):
        img_info = self.data_info[index]

        path = os.path.join(self.img_root, img_info['file_name'])
        img = self.read_image(path)
        w, h = img_info['width'], img_info['height']

        anno = self.read_annotations(index)

        bbox = anno['bboxes']

        bbox_meta = Meta(
            class_id=anno['labels'],
            score=np.ones(len(bbox)).astype(np.float32),
            keep=np.ones(len(bbox)).astype(np.bool_),
        )

        res = dict(
            image=img,
            bbox=bbox,
            bbox_meta=bbox_meta,
            image_meta=dict(ori_size=(w, h), path=path, coco_id=img_info['id']),
        )

        if self.use_keypoint:
            res['point'] = anno['keypoints'][..., :2]
            res['point_meta'] = Meta(keep=anno['keypoints'][..., 2] > 0)

        return res

    def __len__(self):
        return len(self.img_ids)

    def __repr__(self):
        return 'COCOAPIReader(set_path={}, img_root={}, classes={}, use_keypoint={}, {})'.format(self.set, self.img_root, self.classes, self.use_keypoint, super(COCOAPIReader, self).__repr__())
