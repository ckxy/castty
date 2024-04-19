import os
import json
import numpy as np
from .reader import Reader
from .builder import READER
from .utils import read_image_paths
from ..utils.structures import Meta
from ..utils.common import get_image_size
from ...utils.bbox_tools import xyxy2xywh


__all__ = ['PSDParseReader']


def get_rect_points(text_boxes):
    x1 = np.min(text_boxes[:, 0])
    y1 = np.min(text_boxes[:, 1])
    x2 = np.max(text_boxes[:, 2])
    y2 = np.max(text_boxes[:, 3])
    return [x1, y1, x2, y2]


class BoxesConnectorX(object):
    def __init__(self, rects, imageW, max_dist=None, overlap_threshold=None):
        # print('max_dist',max_dist)
        # print('overlap_threshold',overlap_threshold )
        self.rects = np.array(rects)
        self.imageW = imageW
        self.max_dist = max_dist  # x轴方向上合并框阈值
        self.overlap_threshold = overlap_threshold  # y轴方向上最大重合度
        self.graph = np.zeros((self.rects.shape[0], self.rects.shape[0]))  # 构建一个N*N的图 N等于rects的数量

        self.r_index = [[] for _ in range(imageW)]  # 构建imageW个空列表
        for index, rect in enumerate(rects):  # r_index第rect[0]个元素表示 第index个(数量可以是0/1/大于1)rect的x轴起始坐标等于rect[0]
            if int(rect[0]) < imageW:
                self.r_index[int(rect[0])].append(index)
            else:  # 边缘的框旋转后可能坐标越界
                self.r_index[imageW - 1].append(index)
        # print(self.r_index)

    def calc_overlap_for_Yaxis(self, index1, index2):
        # 计算两个框在Y轴方向的重合度(Y轴错位程度)
        height1 = self.rects[index1][3] - self.rects[index1][1]
        height2 = self.rects[index2][3] - self.rects[index2][1]
        y0 = max(self.rects[index1][1], self.rects[index2][1])
        y1 = min(self.rects[index1][3], self.rects[index2][3])
        # print('y1', y1)
        Yaxis_overlap = max(0, y1 - y0) / (max(height1, height2) + 1e-6)

        # print('Yaxis_overlap', Yaxis_overlap)
        return Yaxis_overlap

    def get_proposal(self, index):
        rect = self.rects[index]
        # print('rect',rect)

        for left in range(rect[0] + 1, min(self.imageW - 1, rect[2] + self.max_dist)):
            #print('left',left)
            for idx in self.r_index[left]:
                # print('58796402',idx)
                # index: 第index个rect(被比较rect)
                # idx: 第idx个rect的x轴起始坐标大于被比较rect的x轴起始坐标(+max_dist)且小于被比较rect的x轴终点坐标(+max_dist)
                if self.calc_overlap_for_Yaxis(index, idx) > self.overlap_threshold:

                    return idx

        return -1

    def sub_graphs_connected(self):
        sub_graphs = []       #相当于一个堆栈
        for index in range(self.graph.shape[0]):
            # 第index列全为0且第index行存在非0
            if not self.graph[:, index].any() and self.graph[index, :].any(): #优先级是not > and > or
                v = index
                # print('v',v)
                sub_graphs.append([v])
                # print('sub_graphs', sub_graphs)
                # 级联多个框(大于等于2个)
                # print('self.graph[v, :]', self.graph[v, :])
                while self.graph[v, :].any():

                    v = np.where(self.graph[v, :])[0][0]          #np.where(self.graph[v, :])：(array([5], dtype=int64),)  np.where(self.graph[v, :])[0]：[5]
                    # print('v11',v)
                    sub_graphs[-1].append(v)
                    # print('sub_graphs11', sub_graphs)
        return sub_graphs

    def connect_boxes(self):
        for idx, _ in enumerate(self.rects):

            proposal = self.get_proposal(idx)
            # print('idx11', idx)
            # print('proposal',proposal)
            if proposal >= 0:

                self.graph[idx][proposal] = 1  # 第idx和proposal个框需要合并则置1

        sub_graphs = self.sub_graphs_connected() #sub_graphs [[0, 1], [3, 4, 5]]

        # 不参与合并的框单独存放一个子list
        set_element = set([y for x in sub_graphs for y in x])  #{0, 1, 3, 4, 5}
        for idx, _ in enumerate(self.rects):
            if idx not in set_element:
                sub_graphs.append([idx])            #[[0, 1], [3, 4, 5], [2]]

        result_rects = []
        for sub_graph in sub_graphs:

            rect_set = self.rects[list(sub_graph)]     #[[228  78 238 128],[240  78 258 128]].....
            # print('1234', rect_set)
            rect_set = get_rect_points(rect_set)
            result_rects.append(rect_set)
        return np.array(result_rects)


def get_rect_points(text_boxes):
    x1 = np.min(text_boxes[:, 0])
    y1 = np.min(text_boxes[:, 1])
    x2 = np.max(text_boxes[:, 2])
    y2 = np.max(text_boxes[:, 3])
    return [x1, y1, x2, y2]


class BoxesConnectorY(object):
    def __init__(self, rects, imageH, max_dist=5, overlap_threshold=0.2):
        self.rects = np.array(rects)
        self.imageH = imageH
        self.max_dist = max_dist  # x轴方向上合并框阈值
        self.overlap_threshold = overlap_threshold  # y轴方向上最大重合度
        self.graph = np.zeros((self.rects.shape[0], self.rects.shape[0]))  # 构建一个N*N的图 N等于rects的数量

        self.r_index = [[] for _ in range(imageH)]  # 构建imageH个空列表
        for index, rect in enumerate(rects):  # r_index第rect[0]个元素表示 第index个(数量可以是0/1/大于1)rect的x轴起始坐标等于rect[0]
            if int(rect[1]) < imageH:
                self.r_index[int(rect[1])].append(index)
            else:  # 边缘的框旋转后可能坐标越界
                self.r_index[imageH - 1].append(index)
        # print('self.r_index',self.r_index)
        # print('len(self.r_index)', len(self.r_index))
        # exit()

    def calc_overlap_for_Yaxis(self, index1, index2):
        # 计算两个框在Y轴方向的重合度(Y轴错位程度)
        height1 = self.rects[index1][3] - self.rects[index1][1]
        height2 = self.rects[index2][3] - self.rects[index2][1]
        y0 = max(self.rects[index1][1], self.rects[index2][1])
        y1 = min(self.rects[index1][3], self.rects[index2][3])
        Yaxis_overlap = max(0, y1 - y0) / (max(height1, height2) + 1e-6)

        return Yaxis_overlap

    def calc_overlap_for_Xaxis(self, index1, index2):
        # 计算两个框在Y轴方向的重合度(Y轴错位程度)
        width1 = self.rects[index1][2] - self.rects[index1][0]
        width2 = self.rects[index2][2] - self.rects[index2][0]
        x0 = max(self.rects[index1][0], self.rects[index2][0])
        x1 = min(self.rects[index1][2], self.rects[index2][2])

        Yaxis_overlap = max(0, x1 - x0) / (max(width1, width2) + 1e-6)
        # print('Yaxis_overlap', Yaxis_overlap)
        return Yaxis_overlap

    def get_proposal(self, index):
        rect = self.rects[index]
        for left in range(rect[1] + 1, min(self.imageH - 1, rect[3] + self.max_dist)):
            # print(left, self.r_index[left])
            for idx in self.r_index[left]:
                # print('56871',idx)
                # index: 第index个rect(被比较rect)
                # idx: 第idx个rect的x轴起始坐标大于被比较rect的x轴起始坐标(+max_dist)且小于被比较rect的x轴终点坐标(+max_dist)
                if self.calc_overlap_for_Xaxis(index, idx) > self.overlap_threshold:

                    return idx
        # exit()

        return -1

    def sub_graphs_connected(self):
        sub_graphs = []       #相当于一个堆栈
        for index in range(self.graph.shape[0]):
            # 第index列全为0且第index行存在非0
            if not self.graph[:, index].any() and self.graph[index, :].any(): #优先级是not > and > or
                v = index
                # print('v',v)
                sub_graphs.append([v])
                # print('sub_graphs', sub_graphs)
                # 级联多个框(大于等于2个)
                # print('self.graph[v, :]', self.graph[v, :])
                while self.graph[v, :].any():

                    v = np.where(self.graph[v, :])[0][0]          #np.where(self.graph[v, :])：(array([5], dtype=int64),)  np.where(self.graph[v, :])[0]：[5]
                    # print('v11',v)
                    sub_graphs[-1].append(v)
                    # print('sub_graphs11', sub_graphs)
        return sub_graphs

    def connect_boxes(self):
        for idx, _ in enumerate(self.rects):
            # print('idx', idx)
            proposal = self.get_proposal(idx)

            # print('proposal',proposal)
            if proposal > 0:

                self.graph[idx][proposal] = 1  # 第idx和proposal个框需要合并则置1

        # print(self.graph)
        # exit()

        sub_graphs = self.sub_graphs_connected() #sub_graphs [[0, 1], [3, 4, 5]]

        # 不参与合并的框单独存放一个子list
        set_element = set([y for x in sub_graphs for y in x])  #{0, 1, 3, 4, 5}
        for idx, _ in enumerate(self.rects):
            if idx not in set_element:
                sub_graphs.append([idx])            #[[0, 1], [3, 4, 5], [2]]

        result_rects = []
        for sub_graph in sub_graphs:

            rect_set = self.rects[list(sub_graph)]     #[[228  78 238 128],[240  78 258 128]].....
            # print('1234', rect_set)
            rect_set = get_rect_points(rect_set)
            result_rects.append(rect_set)
        return np.array(result_rects)


@READER.register_module()
class PSDParseReader(Reader):
    def __init__(self, root, **kwargs):
        super(PSDParseReader, self).__init__(**kwargs)

        assert os.path.exists(root)
        self.root = root

        json_paths = self.read_json_paths(root)

        self.json_paths = []
        self.image_paths = []
        self.remove_paths = []

        for json_path in json_paths:
            name = os.path.splitext(os.path.basename(json_path))[0]
            image_path = os.path.join(os.path.dirname(json_path), name + '.jpg')
            remove_path = os.path.join(os.path.dirname(json_path), name + '_remove.jpg')

            if os.path.exists(image_path) and os.path.exists(remove_path):
                self.json_paths.append(json_path)
                self.image_paths.append(image_path)
                self.remove_paths.append(remove_path)

        assert len(self.json_paths) > 0

        self.classes = ['text', 'logo', 'underlay']

        self._info = dict(
            forcat=dict(
                bbox=dict(
                    classes=self.classes,
                    extra_meta=[]
                )
            ), 
            tag_mapping=dict(
                image=['image', 'remove'],
                bbox=['bbox']
            )
        )

    @staticmethod
    def read_json_paths(root_dir):
        jsons = []
        assert os.path.isdir(root_dir), '{}是一个无效的目录'.format(root_dir)

        for root, _, fnames in sorted(os.walk(root_dir)):
            for fname in fnames:
                if fname.lower().endswith('.json'):
                    path = os.path.join(root, fname)
                    jsons.append(path)

        return sorted(jsons)

    @staticmethod
    def iog_calc(boxes1, boxes2):
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        # union_area = boxes1_area + boxes2_area - inter_area
        return inter_area / boxes1_area

    def __getitem__(self, index):
        # index = 24
        # print(index)
        image_path = self.remove_paths[index]

        image = self.read_image(image_path)
        w, h = get_image_size(image)

        with open(self.json_paths[index], 'r') as f:
            data = json.load(f)

        bboxes = []
        for d in data:
            # print(d)
            if d['name'] == 'SY':
                continue

            if d.get('easyocr_bbox', None) is None:
                bboxes.append(d['bbox'])
            else:
                easyocr_bbox = np.array(d['easyocr_bbox']).astype(np.int32)
                easyocr_bbox[..., 0] += d['bbox'][0]
                easyocr_bbox[..., 1] += d['bbox'][1]
                easyocr_bbox[..., 2] += d['bbox'][0]
                easyocr_bbox[..., 3] += d['bbox'][1]
                easyocr_bbox = easyocr_bbox.tolist()
                bboxes.extend(easyocr_bbox)

        # bboxes = np.array(bboxes).astype(np.int32)
        # bboxes[..., ::2] = np.clip(bboxes[..., ::2], 0, w)
        # bboxes[..., 1::2] = np.clip(bboxes[..., 1::2], 0, h)

        # connector_x = BoxesConnectorX(bboxes, w, max_dist=w // 10, overlap_threshold=0.2)
        # bboxes = connector_x.connect_boxes()

        # connector_y = BoxesConnectorY(bboxes, h, max_dist=h // 10, overlap_threshold=0.2)
        # bboxes = connector_y.connect_boxes()

        # if len(bboxes) == 0:
        #     bboxes = np.zeros((0, 4))

        # bboxes_xywh = xyxy2xywh(bboxes)
        # area = bboxes_xywh[..., 2] * bboxes_xywh[..., 3]
        # bboxes = bboxes[area >= 0.005 * w * h]

        # iou = self.iog_calc(bboxes[:, np.newaxis, :], bboxes[np.newaxis, :, :])
        # keep = np.ones(len(bboxes)).astype(np.bool_)

        # mask = iou >= 0.8
        # ind_xs, ind_ys = np.nonzero(mask)
        # for x, y in zip(ind_xs.tolist(), ind_ys.tolist()):
        #     if x == y:
        #         continue

        #     if area[x] <= area[y]:
        #         keep[x] = False
        #     else:
        #         keep[y] = False

        # bboxes = bboxes[keep]

        classes = np.array([0] * len(bboxes)).astype(np.int32)
        bboxes = np.array(bboxes).astype(np.float32)

        bbox_meta = Meta(
            class_id=classes,
            score=np.ones(len(bboxes)).astype(np.float32),
            keep=np.ones(len(bboxes)).astype(np.bool_),
        )

        return dict(
            image=image,
            image_meta=dict(ori_size=(w, h), path=image_path),
            # remove=self.read_image(self.remove_paths[index]),
            # remove_meta=dict(ori_size=(w, h), path=self.remove_paths[index]),
            bbox=bboxes,
            bbox_meta=bbox_meta
        )

    def __len__(self):
        return len(self.json_paths)

    def __repr__(self):
        return f'PSDParseReader(root={self.root}, {super(PSDParseReader, self).__repr__()})'
