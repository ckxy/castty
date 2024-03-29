import cv2
import math
import torch
import numpy as np
from PIL import Image
from .common import is_pil, is_cv2
from ...utils.bbox_tools import xyxy2xywh


def fix_cv2_matrix(M):
    M[0, 2] += (M[0, 0] + M[0, 1] - 1) / 2
    M[1, 2] += (M[1, 0] + M[1, 1] - 1) / 2
    return M


def calc_expand_size_and_matrix(M, img_size):
    w, h = img_size
    P = np.array([(0, 0), (w, 0), (w, h), (0, h)], dtype=np.float32)
    P = warp_point(P, M)

    nw = int(np.ceil(np.max(P[..., 0])) - np.floor(np.min(P[..., 0])))
    nh = int(np.ceil(np.max(P[..., 1])) - np.floor(np.min(P[..., 1])))
    E = np.eye(3)
    E[0, 2] = -np.min(P[..., 0])
    E[1, 2] = -np.min(P[..., 1])
    return E, (nw, nh)


def warp_bbox(bboxes, M):
    if len(bboxes) == 0:
        return bboxes
    # warp points
    xy = np.ones((len(bboxes) * 4, 3))
    xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(len(bboxes) * 4, 2)  # x1y1, x2y2, x1y2, x2y1
    xy = xy @ M.T  # transform
    xy = (xy[:, :2] / xy[:, 2:3]).reshape(len(bboxes), 8)  # rescale
    # create new bboxes
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, len(bboxes)).T
    return xy.astype(np.float32)


def warp_point(points, M):
    if len(points) == 0:
        return points

    xy = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    xy = xy @ M.T  # transform
    xy[:, 0] = xy[:, 0] / xy[:, 2]
    xy[:, 1] = xy[:, 1] / xy[:, 2]
    return xy[:, :2].astype(np.float32)


def warp_image(image, M, dst_size, ccs=False):
    if is_pil(image):
        matrix = np.array(np.matrix(M).I).flatten()
        matrix = (matrix / matrix[-1]).tolist()
        return image.transform(dst_size, Image.PERSPECTIVE, matrix, Image.BILINEAR)
    elif is_cv2(image):
        matrix = np.matrix(M)
        if ccs:
            matrix = fix_cv2_matrix(matrix)
        return cv2.warpPerspective(image, matrix, dst_size)
    else:
        # coeffs = M.flatten().tolist()
        # print(M)
        # matrix = np.array(np.matrix(M).I).flatten()
        # coeffs = (matrix / matrix[-1]).tolist()
        # return perspective(image, dst_size, coeffs, interpolation='bilinear', fill=0)
        raise NotImplementedError('not support')


def warp_mask(mask, M, dst_size, ccs=False):
    if is_cv2(mask):
        matrix = np.matrix(M)
        if ccs:
            matrix = fix_cv2_matrix(matrix)
        mask = cv2.warpPerspective(mask, matrix, dst_size, flags=cv2.INTER_NEAREST)
    else:
        # coeffs = M.flatten().tolist()
        # matrix = np.array(np.matrix(M).I).flatten()
        # coeffs = (matrix / matrix[-1]).tolist()
        # mask = mask.unsqueeze(0)
        # mask = perspective(mask, dst_size, coeffs, interpolation='nearest', fill=0)
        # mask = mask[0]
        raise NotImplementedError('not support')
    return mask


if __name__ == '__main__':
    pass
