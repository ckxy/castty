import cv2
import random
import datetime
import numpy as np
from PIL import Image
from ...utils.bbox_tools import xyxy2xywh
try:
    import pyclipper
except:
    pass


TAG_MAPPING = dict(
    image=['image'],
    label=['label'],
    bbox=['bbox'],
    mask=['mask'],
    point=['point'],
    poly=['poly'],
    seq=['seq']
)


def make_tid():
    return '{0:%Y%m%d%H%M%S%f}'.format(datetime.datetime.now()) + ''.join([str(random.randint(1,10)) for i in range(5)])


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
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(0, width - 1)
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(0, height - 1)
    return bboxes


def clip_point(points, img_size):
    width, height = img_size
    # print(points)
    points[..., 0][points[..., 0] < 0] = -1
    points[..., 0][points[..., 0] >= width] = -1
    points[..., 1][points[..., 1] < 0] = -1
    points[..., 1][points[..., 1] >= height] = -1
    # print(points)
    # exit()
    # points[:, [0, 2]] = bboxes[:, [0, 2]].clip(0, width)
    # points[:, [1, 3]] = bboxes[:, [1, 3]].clip(0, height)
    return points


def clip_poly(polys, img_size):
    width, height = img_size
    subj = (((0, 0), (width, 0), (width, height), (0, height)),)

    tmp = []
    # keep = []
    for i, poly in enumerate(polys):
        wf = np.bitwise_and(poly[..., 0] >= 0, poly[..., 0] < width)
        hf = np.bitwise_and(poly[..., 1] >= 0, poly[..., 1] < height)
        if np.bitwise_and(wf, hf).all():
            tmp.append(poly)
            # keep.append(i)
        else:
            poly_tmp = poly.tolist()
            pc = pyclipper.Pyclipper()
            pc.AddPaths(subj, pyclipper.PT_SUBJECT, True)
            pc.AddPath(poly_tmp, pyclipper.PT_CLIP, True)
            poly_tmp = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
            if len(poly_tmp) > 0:
                tmp.append(np.array(poly_tmp[0]).astype(np.float32))
                # keep.append(i)
            else:
                tmp.append(np.zeros(poly.shape).astype(np.float32) - 1)
    return tmp


def filter_bbox_by_length(bboxes, min_w=1, min_h=1):
    # w = (bboxes[..., 2] - bboxes[..., 0]) > 1
    # h = (bboxes[..., 3] - bboxes[..., 1]) > 1
    # t = np.logical_and(w, h)
    # keep = np.nonzero(t)[0].tolist()
    xywh = xyxy2xywh(bboxes.copy())
    keep = (xywh[..., 2] >= min_w) * (xywh[..., 3] >= min_h)
    # keep = np.nonzero(keep)[0].tolist()
    return keep


def filter_bbox_by_center(bboxes, img_size):
    width, height = img_size
    bboxes = xyxy2xywh(bboxes)
    centers = bboxes[..., :2]
    # t = np.logical_and(centers[..., 0] >= 0, centers[..., 0] < width)
    # s = np.logical_and(centers[..., 1] >= 0, centers[..., 1] < height)
    # keep = np.logical_and(t, s)
    keep = (centers[..., 0] >= 0 & centers[..., 0] < width) & (centers[..., 1] >= 0 & centers[..., 1] < height)
    keep = np.nonzero(keep)[0].tolist()
    return keep


# def filter_point(points, img_size):
#     width, height = img_size
#     x = (points[:, 0] >= 0) & (points[:, 0] < width)
#     y = (points[:, 1] >= 0) & (points[:, 1] < height)
#     discard = np.nonzero(~(x & y))[0]
#     return discard


def filter_point(points):
    x = points[..., 0] < 0
    y = points[..., 1] < 0
    res = ~(x | y)
    count = res.astype(np.int32).sum(1) > 0
    return res, count


def filter_poly(polys):
    res = []

    for poly in polys:
        x = poly[..., 0] < 0
        y = poly[..., 1] < 0
        res.append(~(x | y).any())
    return np.array(res)


def filter_list(l, keep):
    assert len(l) == len(keep)
    res = []

    for i, flag in enumerate(keep):
        if flag:
            res.append(l[i])
    return res

