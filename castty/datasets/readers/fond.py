import numpy as np
from PIL import Image, ImageDraw

from .reader import Reader
from .builder import READER
from ..utils.structures import Meta
from ..utils.common import get_image_size


__all__ = ['FondReader']


@READER.register_module()
class FondReader(Reader):
    def __init__(self, mode, image, label, bbox, mask, point, poly, **kwargs):
        super(FondReader, self).__init__(**kwargs)

        self.mode = mode
        self.image = image
        self.label = label
        self.bbox = bbox
        self.mask = mask
        self.point = point
        self.poly = poly

        self._info = dict(forcat=dict(), tag_mapping=dict(image=['image']))

        if 'label' in self.mode:
            self._info['forcat']['label'] = dict(
                classes=[
                    ['normal pose', 'large pose'], 
                    ['normal expression', 'exaggerate expression'], 
                    ['normal illumination', 'extreme illumination'], 
                    ['no make-up', 'make-up'],
                    ['no occlusion', 'occlusion'],
                    ['clear', 'blur']
                ]
            )
            self._info['tag_mapping']['label'] = ['label']
        if 'bbox' in self.mode:
            self._info['forcat']['bbox'] = dict(classes=['face'])
            self._info['tag_mapping']['bbox'] = ['bbox']
        if 'mask' in self.mode:
            self._info['forcat']['mask'] = dict(classes=['__background__', 'eye'])
            self._info['tag_mapping']['mask'] = ['mask']
        if 'point' in self.mode:
            self._info['forcat']['point'] = dict(classes=[str(i) for i in range(5)])
            self._info['tag_mapping']['point'] = ['point']
        if 'poly' in self.mode:
            self._info['forcat']['poly'] = dict(classes=['eye'])
            self._info['tag_mapping']['poly'] = ['poly']

    def __getitem__(self, index):
        image = self.read_image(self.image)
        w, h = get_image_size(image)

        labels = [np.array(i, dtype=np.float32) for i in self.label]

        bboxes = np.array(self.bbox).astype(np.float32).reshape(-1, 4)
        bbox_meta = Meta(
            class_id=np.zeros(len(bboxes)).astype(np.int32),
            score=np.ones(len(bboxes)).astype(np.float32),
            keep=np.ones(len(bboxes)).astype(np.bool_),
        )

        mask = Image.new('P', (w, h), 0)
        pts = np.array(self.mask).astype(np.int32).flatten().tolist()
        ImageDraw.Draw(mask).polygon(pts, fill=1)
        mask = np.array(mask).astype(np.int32)

        points = np.array(self.point).astype(np.float32).reshape(1, -1, 2)
        point_meta = Meta(keep=np.ones(points.shape[:2]).astype(np.bool_))

        polys = [np.array(self.poly).astype(np.float32)]
        poly_meta = Meta(
            class_id=np.zeros(len(polys)).astype(np.int32),
            keep=np.ones(len(polys)).astype(np.bool_),
        )

        res = dict(
            image=image,
            image_meta=dict(ori_size=(w, h), path=self.image)
        )

        if 'label' in self.mode:
            res['label'] = labels
        if 'bbox' in self.mode:
            res['bbox'] = bboxes
            res['bbox_meta'] = bbox_meta
        if 'mask' in self.mode:
            res['mask'] = mask
            # res['mask_meta'] = dict(ori_size=(w, h))
        if 'point' in self.mode:
            res['point'] = points
            res['point_meta'] = point_meta
        if 'poly' in self.mode:
            res['poly'] = polys
            res['poly_meta'] = poly_meta

        return res

    def __len__(self):
        return 1

    def __repr__(self):
        return 'FondReader(mode={}, {})'.format(self.mode, super(FondReader, self).__repr__())
