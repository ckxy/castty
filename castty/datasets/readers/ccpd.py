import os
import numpy as np
from .reader import Reader
from .builder import READER
from .utils import is_image_file, read_image_paths
from ..utils.structures import Meta
from ..utils.common import get_image_size


__all__ = ['CCPDFolderReader']


provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


@READER.register_module()
class CCPDFolderReader(Reader):
    def __init__(self, root, need_extra_info=False, **kwargs):
        super(CCPDFolderReader, self).__init__(**kwargs)

        self.root = root
        self.need_extra_info = need_extra_info
        self.image_paths = read_image_paths(self.root)

        assert len(self.image_paths) > 0

        self._info = dict(
            forcat=dict(
                bbox=dict(
                    classes=['carplate']
                ),
                point=dict(
                    classes=[str(i) for i in range(4)],
                )
            ),
            tag_mapping=dict(
                image=['image'],
                bbox=['bbox'],
                point=['point'],
                seq=['seq']
            )
        )

    def __getitem__(self, index):
        img = self.read_image(self.image_paths[index])
        w, h = get_image_size(img)

        name = os.path.splitext(os.path.basename(self.image_paths[index]))[0]
        area, tilt_degrees, box, points, text, brightness, blurriness = name.split('-')

        box = [list(map(int, i.split('&'))) for i in box.split('_')]
        box = np.array(box).astype(np.float32)
        box = box.flatten()[np.newaxis, ...]
        bbox_meta = Meta(
            class_id=np.zeros(len(box)).astype(np.int32),
            score=np.ones(len(box)).astype(np.float32),
            keep=np.ones(len(box)).astype(np.bool_),
        )

        points = [list(map(int, i.split('&'))) for i in points.split('_')]
        points = np.array(points).astype(np.float32)
        points = points[np.newaxis, ...]
        point_meta = Meta(keep=np.ones(points.shape[:2]).astype(np.bool_))

        text = text.split('_')
        text[0] = provinces[int(text[0])]
        text[1] = alphabets[int(text[1])]
        for i in range(2, len(text)):
            text[i] = ads[int(text[i])]
        text = ''.join(text)

        image_meta = dict(ori_size=(w, h), path=self.image_paths[index])
        if self.need_extra_info:
            image_meta['area'] = int(area) / 100
            image_meta['tilt_degrees'] = tuple([int(d) for d in tilt_degrees.split('_')])
            image_meta['brightness'] = int(brightness)
            image_meta['blurriness'] = int(blurriness)

        return dict(
            image=img,
            image_meta=image_meta,
            bbox=box,
            bbox_meta=bbox_meta,
            point=points,
            point_meta=point_meta,
            seq=text
        )

    def __len__(self):
        return len(self.image_paths)

    def __repr__(self):
        return 'CCPDFolderReader(root={}, need_extra_info={}, {})'.format(self.root, self.need_extra_info, super(CCPDFolderReader, self).__repr__())
