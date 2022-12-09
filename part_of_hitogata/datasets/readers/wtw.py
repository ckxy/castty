import os
import copy
import numpy as np
from .reader import Reader
from .builder import READER
from ..utils.structures import Meta
from ..utils.common import get_image_size
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from ..bamboo.crop import crop_image, crop_bbox, crop_point
from PIL import Image, ImageDraw
import math


__all__ = ['WTWReader', 'WTWLineReader', 'WTWSTReader']


@READER.register_module()
class WTWReader(Reader):
    def __init__(self, root, **kwargs):
        super(WTWReader, self).__init__(**kwargs)

        assert os.path.exists(os.path.join(root, 'images'))
        assert os.path.exists(os.path.join(root, 'xml'))

        self.root = root
        self.xml_paths = sorted(os.listdir(os.path.join(root, 'xml')))

        self._info = dict(
            forcat=dict(
                type='kpt-det',
                bbox_classes=['grid']
            ),
        )

    def read_annotations(self, xml_path):
        root = ET.parse(xml_path).getroot()
        objects = root.findall('object')

        bboxes = []
        points = []
        table_ids = []
        startcols = []
        endcols = []
        startrows = []
        endrows = []

        for obj in objects:
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text.strip())
            xmax = float(bbox.find('xmax').text.strip())
            ymin = float(bbox.find('ymin').text.strip())
            ymax = float(bbox.find('ymax').text.strip())

            point = []
            for i in range(1, 5):
                x = float(bbox.find('x{}'.format(i)).text.strip())
                y = float(bbox.find('y{}'.format(i)).text.strip())
                point.append([x, y])

            bboxes.append([xmin, ymin, xmax, ymax])
            points.append(point)

            table_ids.append(int(bbox.find('tableid').text.strip()))
            startcols.append(int(bbox.find('startcol').text.strip()))
            endcols.append(int(bbox.find('endcol').text.strip()))
            startrows.append(int(bbox.find('startrow').text.strip()))
            endrows.append(int(bbox.find('endrow').text.strip()))

        bboxes = np.array(bboxes, dtype=np.float32)
        points = np.array(points, dtype=np.float32)
        table_ids = np.array(table_ids, dtype=np.int32)
        startcols = np.array(startcols, dtype=np.int32)
        endcols = np.array(endcols, dtype=np.int32)
        startrows = np.array(startrows, dtype=np.int32)
        endrows = np.array(endrows, dtype=np.int32)

        return bboxes, points, table_ids, startcols, endcols, startrows, endrows

    def __call__(self, index):
        # index = 4
        index = 159
        path = os.path.join(self.root, 'images', os.path.splitext(self.xml_paths[index])[0] + '.jpg')
        img = self.read_image(path)
        w, h = get_image_size(img)

        bboxes, points, table_ids, startcols, endcols, startrows, endrows = self.read_annotations(os.path.join(self.root, 'xml', self.xml_paths[index]))

        # if len(np.unique(table_ids)) > 2:
        #     print(index)

        bbox_meta = Meta(
            class_id=np.zeros(len(bboxes)).astype(np.int32),
            score=np.ones(len(bboxes)).astype(np.float32),
            table_id=table_ids,
            startcol=startcols,
            endcol=endcols,
            startrow=startrows,
            endrow=endrows
        )

        point_meta = Meta(visible=np.ones(points.shape[:2]).astype(np.bool))

        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=path,
            bbox=bboxes,
            bbox_meta=bbox_meta,
            point=points,
            point_meta=point_meta
        )

    def __len__(self):
        return len(self.xml_paths)

    def __repr__(self):
        return 'WTWReader(root={}, {})'.format(self.root, super(WTWReader, self).__repr__())


@READER.register_module()
class WTWSTReader(WTWReader):
    def __init__(self, root, **kwargs):
        super(WTWSTReader, self).__init__(root, **kwargs)

        from tqdm import tqdm
        import torch

        # self.data_lines = []
        # for i in tqdm(range(len(self.xml_paths))):
        #     table_ids = self.read_table_ids(os.path.join(self.root, 'xml', self.xml_paths[i]))
        #     for j in range(len(np.unique(table_ids))):
        #         self.data_lines.append((self.xml_paths[i], j))

        # torch.save(self.data_lines, 'wtwst.pth')
        self.data_lines = torch.load('wtwst.pth')

    def read_table_ids(self, xml_path):
        root = ET.parse(xml_path).getroot()
        objects = root.findall('object')

        table_ids = []
        for obj in objects:
            bbox = obj.find('bndbox')
            table_ids.append(int(bbox.find('tableid').text.strip()))
        return np.array(table_ids, dtype=np.int32)

    def calc_cropping(self, bboxes, table_ids, table_id, size):
        mask = Image.new('P', size, 0)
        w, h = size

        ind = table_ids == table_id

        x1 = int(math.floor(np.min(bboxes[ind][..., 0])))
        y1 = int(math.floor(np.min(bboxes[ind][..., 1])))
        x2 = int(math.ceil(np.max(bboxes[ind][..., 2])))
        y2 = int(math.ceil(np.max(bboxes[ind][..., 3])))

        for bbox, tid in zip(bboxes, table_ids):
            draw = ImageDraw.Draw(mask)
            if tid != table_id:
                draw.rectangle(tuple(bbox.astype(np.int32).tolist()), fill=1)
        
        mask = np.asarray(mask)

        w_array = mask[y1: y2, :]
        w_array = w_array.sum(axis=0)

        h_array = mask[:, x1: x2]
        h_array = h_array.sum(axis=1)

        left = 0
        for i in range(x1 - 1, -1, -1):
            if w_array[i] > 0:
                break
            left += 1
        left = x1 - left + 1 if left > 0 else x1

        right = 0
        for i in range(x2 + 1, w):
            if w_array[i] > 0:
                break
            right += 1
        right = x2 + right - 1 if right > 0 else x2

        top = 0
        for i in range(y1 - 1, -1, -1):
            if h_array[i] > 0:
                break
            top += 1
        top = y1 - top + 1 if top > 0 else y1

        bottom = 0
        for i in range(y2 + 1, h):
            if h_array[i] > 0:
                break
            bottom += 1
        bottom = y1 + bottom - 1 if bottom > 0 else y2

        return left, top, right, bottom

    def __call__(self, index):
        index = 1
        xml_path, table_id = self.data_lines[index]

        path = os.path.join(self.root, 'images', os.path.splitext(xml_path)[0] + '.jpg')
        img = self.read_image(path)
        w, h = get_image_size(img)

        bboxes, points, table_ids, startcols, endcols, startrows, endrows = self.read_annotations(os.path.join(self.root, 'xml', xml_path))

        if len(np.unique(table_ids)) > 1:
            left, top, right, bottom = self.calc_cropping(bboxes, table_ids, table_id, get_image_size(img))

            ind = table_ids == table_id

            bboxes = bboxes[ind]
            points = points[ind]

            startcols = startcols[ind]
            endcols = endcols[ind]
            startrows = startrows[ind]
            endrows = endrows[ind]
            
            img = crop_image(img, left, top, right, bottom)
            bboxes = crop_bbox(bboxes, left, top)
            points = crop_point(points, left, top)

        bbox_meta = Meta(
            class_id=np.zeros(len(bboxes)).astype(np.int32),
            score=np.ones(len(bboxes)).astype(np.float32),
            startcol=startcols,
            endcol=endcols,
            startrow=startrows,
            endrow=endrows
        )

        point_meta = Meta(visible=np.ones(points.shape[:2]).astype(np.bool))

        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=path,
            bbox=bboxes,
            bbox_meta=bbox_meta,
            point=points,
            point_meta=point_meta
        )

    def __len__(self):
        return len(self.data_lines)

    def __repr__(self):
        return 'WTWSTReader(root={}, {})'.format(self.root, super(WTWReader, self).__repr__())


@READER.register_module()
class WTWLineReader(WTWReader):
    def __init__(self, root, **kwargs):
        super(WTWLineReader, self).__init__(root, **kwargs)

        self._info = dict(
            forcat=dict(
                type='ocrdet',
            ),
        )

    def __call__(self, index):
        # index = 235
        path = os.path.join(self.root, 'images', os.path.splitext(self.xml_paths[index])[0] + '.jpg')
        img = self.read_image(path)
        w, h = get_image_size(img)

        bboxes, points, table_ids, startcols, endcols, startrows, endrows = self.read_annotations(os.path.join(self.root, 'xml', self.xml_paths[index]))

        polys = []
        for i in range(len(points)):
            polys.append(points[i])

        meta = Meta(ignore_flag=np.zeros(len(polys)).astype(np.bool))

        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=path,
            poly=polys,
            poly_meta=meta
        )

    def __repr__(self):
        return 'WTWLineReader(root={}, {})'.format(self.root, super(WTWReader, self).__repr__())
