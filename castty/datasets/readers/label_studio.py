import os
import cv2
import json
import numpy as np
from .reader import Reader
from .builder import READER
from PIL import Image, ImageDraw
from ..utils.common import get_image_size


__all__ = ['LSSSwithPolygonsReader']


def maximum_internal_rectangle(img_bin): 
    contours, _ = cv2.findContours(img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
 
    contour = contours[0].reshape(len(contours[0]), 2)
 
    rect = []
 
    for i in range(len(contour)):
        x1, y1 = contour[i]
        for j in range(len(contour)):
            x2, y2 = contour[j]
            area = abs(y2 - y1) * abs(x2 - x1)
            rect.append(((x1, y1), (x2, y2), area))
 
    all_rect = sorted(rect, key=lambda x: x[2], reverse=True)
 
    if all_rect:
        best_rect_found = False
        index_rect = 0
        nb_rect = len(all_rect)
 
        while not best_rect_found and index_rect < nb_rect:
 
            rect = all_rect[index_rect]
            (x1, y1) = rect[0]
            (x2, y2) = rect[1]
 
            valid_rect = True
 
            x = min(x1, x2)
            while x < max(x1, x2) + 1 and valid_rect:
                if any(img_bin[y1, x]) == 0 or any(img_bin[y2, x]) == 0:
                    valid_rect = False
                x += 1
 
            y = min(y1, y2)
            while y < max(y1, y2) + 1 and valid_rect:
                if any(img_bin[y, x1]) == 0 or any(img_bin[y, x2]) == 0:
                    valid_rect = False
                y += 1
 
            if valid_rect:
                best_rect_found = True
 
            index_rect += 1
 
        if best_rect_found:
            # 如果要在灰度图img_gray上画矩形，请用黑色画（0,0,0）
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            cv2.imshow("rec", img)
            cv2.waitKey(0)
 
        else:
            print("No rectangle fitting into the area")
 
    else:
        print("No rectangle found")


@READER.register_module()
class LSSSwithPolygonsReader(Reader):
    def __init__(self, root, set_path, classes, **kwargs):
        super(LSSSwithPolygonsReader, self).__init__(**kwargs)

        assert os.path.exists(root)
        assert os.path.exists(set_path)
        assert classes[0] == '__background__'

        self.root = root
        self.set_path = set_path
        self.classes = classes

        with open(set_path, 'r') as f:
            self.data = json.load(f)

        assert len(self.data) > 0

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
        data = self.data[index]

        name = os.path.basename(data['data']['image']).split('-')[1:]
        name = '-'.join(name)
        path = os.path.join(self.root, name)
        img = self.read_image(path)

        w, h = get_image_size(img)
        mask = Image.new('P', (w, h), 0)

        for annotation in data['annotations']:
            for result in annotation['result']:
                if result['type'] != 'polygonlabels':
                    continue

                value = result['value']
                polygonlabels = value.get('polygonlabels', [])

                if len(polygonlabels) == 0 or polygonlabels[0].lower() not in self.classes:
                    continue

                pts = np.array(value['points']) / 100
                pts[..., 0] *= w
                pts[..., 1] *= h
                pts = pts.astype(np.int32).flatten().tolist()
                ImageDraw.Draw(mask).polygon(pts, fill=self.classes.index(polygonlabels[0].lower()))

        mask = np.array(mask).astype(np.int32)

        mask[mask == 1] = 255
        mask = mask.astype(np.uint8)
        # print(np.unique(mask))
        # maximum_internal_rectangle(mask)
        # exit()

        return dict(
            image=img,
            image_meta=dict(ori_size=(w, h), path=path),
            mask=mask,
            mask_meta=dict(ori_size=(w, h))
        )

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return 'LabelmeMaskReader(root={}, classes={}, {})'.format(self.root, self.classes, super(LSSSwithPolygonsReader, self).__repr__())
