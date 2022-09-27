import cv2
import numpy as np
from PIL import Image
from ...utils.bbox_tools import xyxy2xywh
try:
    import pyclipper
except:
    pass


def is_pil(img):
    if isinstance(img, Image.Image):
        return True
    else:
        return False


def get_image_size(img):
    if isinstance(img, Image.Image):
        w, h = img.size
    elif isinstance(img, np.ndarray):
        if len(img.shape) == 3:
            h, w, _ = img.shape
        else:
            h, w = img.shape
    else:
        _, h, w = img.shape
    return w, h


def clip_bbox(bboxes, img_size):
    width, height = img_size
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(0, width)
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(0, height)
    return bboxes


def clip_poly(polys, img_size):
    width, height = img_size
    subj = (((0, 0), (width, 0), (width, height), (0, height)),)

    tmp = []
    keep = []
    for i, poly in enumerate(polys):
        wf = np.bitwise_and(poly[..., 0] >= 0, poly[..., 0] < width)
        hf = np.bitwise_and(poly[..., 1] >= 0, poly[..., 1] < height)
        if np.bitwise_and(wf, hf).all():
            tmp.append(poly)
            keep.append(i)
        else:
            poly = poly.tolist()
            pc = pyclipper.Pyclipper()
            pc.AddPaths(subj, pyclipper.PT_SUBJECT, True)
            pc.AddPath(poly, pyclipper.PT_CLIP, True)
            poly = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
            if len(poly) > 0:
                tmp.append(np.array(poly[0]).astype(np.float32))
                keep.append(i)
    return tmp, keep


def filter_bbox(bboxes):
    w = (bboxes[..., 2] - bboxes[..., 0]) > 1
    h = (bboxes[..., 3] - bboxes[..., 1]) > 1
    t = np.logical_and(w, h)
    keep = np.nonzero(t)[0].tolist()
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
    width, height = img_size
    x = (points[:, 0] >= 0) & (points[:, 0] < width)
    y = (points[:, 1] >= 0) & (points[:, 1] < height)
    discard = np.nonzero(~(x & y))[0]
    return discard
