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
                type='det-kpt',
            )
        )

    def __call__(self, index):
        line = self.data_lines[index].strip().split()
        landmark = np.asarray(list(map(float, line[:196])), dtype=np.float32).reshape(1, -1, 2)
        box = np.asarray(list(map(int, line[196:200])))
        attribute = np.asarray(list(map(int, line[200:206])), dtype=np.int)
        name = line[206]

        img = self.read_image(os.path.join(self.img_root, name))
        w, h = get_image_size(img)

        point_meta = Meta(visible=np.ones(landmarks.shape[:2]).astype(np.bool))
        # meta = Meta(['visible'], [np.ones(landmark.shape[:2]).astype(np.bool)])
        # print(meta.get('visible').shape, landmark.shape)
        # exit()
        
        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=os.path.join(self.img_root, name),
            bbox=box[np.newaxis, ...].astype(np.float32),
            point=landmark,
            point_meta=meta,
            # attribute=attribute
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

        # for i, name in enumerate(self.names):
        #     if len(self.data[name]) > 1:
        #         print(i)

        assert len(self.names) > 0

        self._info = dict(
            forcat=dict(
                type='det-kpt',
            )
        )

    def __call__(self, index):
        # index = 4993
        name = self.names[index]
        lines = self.data[name]

        landmarks = []
        boxes = []

        for line in lines:
            line = line.split()
            landmark = np.asarray(list(map(float, line[:196])), dtype=np.float32).reshape(-1, 2)
            box = np.asarray(list(map(int, line[196:200]))).astype(np.float32)

            landmarks.append(landmark)
            boxes.append(box)

        landmarks = np.asarray(landmarks)
        boxes = np.asarray(boxes)

        img = self.read_image(os.path.join(self.img_root, name))
        w, h = get_image_size(img)

        point_meta = Meta(visible=np.ones(landmarks.shape[:2]).astype(np.bool))
        bbox_meta = Meta(box2point=np.arange(len(landmarks)))
        
        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=os.path.join(self.img_root, name),
            bbox=boxes,
            point=landmarks,
            point_meta=point_meta,
            bbox_meta=bbox_meta,
            # attribute=attribute
        )

    def __len__(self):
        return len(self.names)

    def __repr__(self):
        return 'WFLWSIReader(txt={}, {})'.format(self.txt, super(WFLWSIReader, self).__repr__())

