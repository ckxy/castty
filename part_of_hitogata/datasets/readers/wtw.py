import os
import numpy as np
from .reader import Reader
from .builder import READER
from ..utils.structures import Meta
from ..utils.common import get_image_size
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


__all__ = ['WTWReader']


@READER.register_module()
class WTWReader(Reader):
    def __init__(self, root, **kwargs):
        super(WTWReader, self).__init__(**kwargs)

        assert os.path.exists(os.path.join(root, 'images'))
        assert os.path.exists(os.path.join(root, 'xml'))

        self.root = root
        a = os.listdir(os.path.join(root, 'images'))
        # print(set([os.path.splitext(os.path.basename(aa))[1] for aa in a]))
        # for aa in a:
        #     if os.path.splitext(os.path.basename(aa))[1] != '.jpg':
        #         print(aa)
        b = os.listdir(os.path.join(root, 'xml'))
        # print(len(a), len(b))
        self.xml_paths = sorted(os.listdir(os.path.join(root, 'xml')))

        # for x in self.xml_paths:
        #     if x.startswith('002383'):
        #         print(x, self.xml_paths.index(x))
        #         exit()
        # print(set([os.path.splitext(os.path.basename(aa))[1] for aa in b]))
        # exit()

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
        # print(bboxes.shape, points.shape, table_ids.shape, startcols.shape, endcols.shape, startrows.shape, endrows.shape)
        # exit()
        return bboxes, points, table_ids, startcols, endcols, startrows, endrows

    def __call__(self, index):
        # index = 4
        path = os.path.join(self.root, 'images', os.path.splitext(self.xml_paths[index])[0] + '.jpg')
        img = self.read_image(path)
        w, h = get_image_size(img)

        bboxes, points, table_ids, startcols, endcols, startrows, endrows = self.read_annotations(os.path.join(self.root, 'xml', self.xml_paths[index]))

        # if (table_ids > 0).any():
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
