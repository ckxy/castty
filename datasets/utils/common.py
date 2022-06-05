import cv2
import numpy as np
from PIL import Image
from utils.bbox_tools import xyxy2xywh


def is_pil(img):
    if isinstance(img, Image.Image):
        return True
    else:
        return False


def get_image_size(img):
    if isinstance(img, Image.Image):
        w, h = img.size
    else:
        if len(img.shape) == 3:
            h, w, _ = img.shape
        else:
            h, w = img.shape
    return w, h


def clip_bbox(bboxes, img_size):
    width, height = img_size
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(0, width)
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(0, height)
    return bboxes


def filter_bbox(bboxes):
    n = len(bboxes)
    keep = []
    for i in range(n):
        x1, y1, x2, y2 = bboxes[i]
        if x2 - x1 > 1 and y2 - y1 > 1:
            keep.append(i)
    return keep


def filter_bbox_by_center(bboxes, img_size):
    width, height = img_size
    bboxes = xyxy2xywh(bboxes)
    centers = bboxes[..., :2]
    t = np.logical_and(centers[..., 0] >= 0, centers[..., 0] < width)
    s = np.logical_and(centers[..., 1] >= 0, centers[..., 1] < height)
    keep = np.logical_and(t, s)
    keep = np.nonzero(keep)[0].tolist()
    return keep


def filter_point(points, img_size):
    n = len(points)
    width, height = img_size
    x = (points[:, 0] >= 0) & (points[:, 0] < width)
    y = (points[:, 1] >= 0) & (points[:, 1] < height)
    discard = np.nonzero(~(x & y))[0]
    return discard
