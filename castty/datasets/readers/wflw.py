import os
import numpy as np
from .reader import Reader
from .builder import READER
from ..utils.structures import Meta
from ..utils.common import get_image_size


__all__ = ['WFLWReader', 'WFLWSIReader']


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

        self._info = dict(
            forcat=dict(
                bbox=dict(
                    classes=['face'],
                ),
                point=dict(
                    classes=[str(i) for i in range(98)],
                ),
            ),
            tag_mapping=dict(
                image=['image'],
                bbox=['bbox'],
                point=['point']
            )
        )

    def __getitem__(self, index):
        line = self.data_lines[index].strip().split()
        landmark = np.array(list(map(float, line[:196])), dtype=np.float32).reshape(1, -1, 2)
        box = np.array(list(map(int, line[196:200])))
        attribute = np.array(list(map(int, line[200:206])), dtype=np.int32)
        name = line[206]

        img = self.read_image(os.path.join(self.img_root, name))
        w, h = get_image_size(img)

        point_meta = Meta(keep=np.ones(landmark.shape[:2]).astype(np.bool_))
        bbox_meta = Meta(
            class_id=np.zeros([1]).astype(np.int32),
            score=np.ones([1]).astype(np.float32),
            keep=np.ones([1]).astype(np.bool_),
            box2point=np.zeros([1]).astype(np.int32),
        )
        
        return dict(
            image=img,
            bbox=box[np.newaxis, ...].astype(np.float32),
            point=landmark,
            point_meta=point_meta,
            bbox_meta=bbox_meta,
            image_meta=dict(ori_size=(w, h), path=os.path.join(self.img_root, name)),
        )

    def __len__(self):
        return len(self.data_lines)

    def __repr__(self):
        return 'WFLWReader(txt={}, {})'.format(self.txt, super(WFLWReader, self).__repr__())


@READER.register_module()
class WFLWSIReader(Reader):
    def __init__(self, root, txt_path, **kwargs):
        super(WFLWSIReader, self).__init__(**kwargs)

        self.root = root
        self.txt = txt_path
        self.img_root = os.path.join(self.root, 'WFLW_images')
        assert os.path.exists(self.img_root)

        with open(os.path.join(self.root, self.txt), 'r') as f:
            data_lines = f.readlines()

        self.data = dict()
        for line in data_lines:
            name = line.strip().split()[206]
            if name not in self.data.keys():
                self.data[name] = [line.strip()]
            else:
                self.data[name].append(line.strip())
        self.names = sorted(list(self.data.keys()))

        assert len(self.names) > 0

        self._info = dict(
            forcat=dict(
                bbox=dict(
                    classes=['face'],
                ),
                point=dict(
                    classes=[str(i) for i in range(98)],
                ),
            ),
            tag_mapping=dict(
                image=['image'],
                bbox=['bbox'],
                point=['point']
            )
        )

    def __getitem__(self, index):
        # index = 4993
        name = self.names[index]
        lines = self.data[name]

        landmarks = []
        boxes = []

        for line in lines:
            line = line.split()
            landmark = np.array(list(map(float, line[:196])), dtype=np.float32).reshape(-1, 2)
            box = np.array(list(map(int, line[196:200]))).astype(np.float32)

            landmarks.append(landmark)
            boxes.append(box)

        landmarks = np.array(landmarks)
        boxes = np.array(boxes)

        img = self.read_image(os.path.join(self.img_root, name))
        w, h = get_image_size(img)

        point_meta = Meta(keep=np.ones(landmarks.shape[:2]).astype(np.bool_))

        bbox_meta = Meta(
            class_id=np.zeros(len(landmarks)).astype(np.int32),
            score=np.ones(len(landmarks)).astype(np.float32),
            keep=np.ones(len(landmarks)).astype(np.bool_),
            box2point=np.arange(len(landmarks)).astype(np.int32),
        )
        
        return dict(
            image=img,
            bbox=boxes,
            point=landmarks,
            point_meta=point_meta,
            bbox_meta=bbox_meta,
            image_meta=dict(ori_size=(w, h), path=os.path.join(self.img_root, name)),
        )

    def __len__(self):
        return len(self.names)

    def __repr__(self):
        return 'WFLWSIReader(txt={}, {})'.format(self.txt, super(WFLWSIReader, self).__repr__())

