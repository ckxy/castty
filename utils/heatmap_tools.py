import cv2
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def calc_gaussian_2d(sigma=1, step=0.01, alpha=True):
    # sigma = 1
    radius = 3 * sigma
    if alpha:
        a = 1 / (2 * np.pi * (sigma ** 2))
    else:
        a = 1

    x, y = np.meshgrid(np.arange(-radius, radius + step, step), np.arange(-radius, radius + step, step))
    d = np.power(x, 2) + np.power(y, 2)
    gaussian_map = a * np.exp(-0.5 * d / (sigma ** 2))
    return gaussian_map.astype(np.float32)


def draw_heatmap(heatmap):
    if isinstance(heatmap, np.ndarray):
        hm = heatmap
    else:
        hm = heatmap.detach().cpu().numpy()
    assert np.max(hm) <= 1 and np.min(hm) >= 0
    res = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    res = res[..., ::-1]
    return Image.fromarray(res)


def minmax_norm(heatmap):
    minv = np.min(heatmap)
    maxv = np.max(heatmap)
    return (heatmap.copy() - minv) / (maxv - minv)


def heatmap2quad(src, mode=0, ori_img=None):
    """
    Performs a marker-based image segmentation using the watershed algorithm.
    :param src: 8-bit 1-channel image.
    :return: 32-bit single-channel image (map) of markers.
    """
    gray = src.copy()
    gray = np.uint8(np.clip(gray, 0, 1) * 255)

    if mode == 0:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(gray, 0.2 * 255, 255, cv2.THRESH_BINARY)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    if mode == 0:
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
    else:
        sure_bg = opening

    # Finding sure foreground area
    if mode == 0:
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, cv2.THRESH_BINARY)
    else:
        _, sure_fg = cv2.threshold(gray, 0.6 * 255, 255, cv2.THRESH_BINARY)

    # Finding unknown region
    # sure_bg = np.uint8(sure_bg)
    sure_fg = np.uint8(sure_fg)
    # cv2.imshow('sure_fg', sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker label
    lingret, marker_map = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    marker_map = marker_map + 1

    # Now, mark the region of unknown with zero
    marker_map[unknown == 255] = 0

    # marker_map = cv2.watershed(img, marker_map)

    boxes = []
    marker_count = np.max(marker_map)
    for marker_number in range(2, marker_count + 1):
        # y, x = np.array(np.where(marker_map == marker_number))
        # xmin = np.min(x)
        # xmax = np.max(x) + 1
        # ymin = np.min(y)
        # ymax = np.max(y) + 1
        # box = [xmin, ymin, xmax, ymax]

        # area = (xmax - xmin) * (ymax - ymin)
        # if area < 10:
        #     continue
        marker_cnt = np.swapaxes(np.array(np.where(marker_map == marker_number)), axis1=0, axis2=1)[:, ::-1]
        rect = cv2.minAreaRect(marker_cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # box = np.array(box)
        boxes.append(box)

    if ori_img is not None:
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        vis_img = img.copy()
        vis_img = np.vstack([vis_img, cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)])

        vis_img = np.vstack([vis_img, cv2.cvtColor(sure_bg, cv2.COLOR_GRAY2BGR)])
        vis_img = np.vstack([vis_img, cv2.cvtColor(sure_fg, cv2.COLOR_GRAY2BGR)])
        vis_img = np.vstack([vis_img, cv2.cvtColor(unknown, cv2.COLOR_GRAY2BGR)])

        color_markers = np.uint8(marker_map + 1)
        color_markers = color_markers / (color_markers.max() / 255)
        color_markers = np.uint8(color_markers)
        color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)
        vis_img = np.vstack([vis_img, color_markers])

        tmp = ori_img.copy()
        tmp[marker_map == -1] = [255, 255, 0]
        vis_img = np.vstack([vis_img, tmp])

        tmp = ori_img.copy()
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            # cv2.rectangle(tmp, (xmin, ymin), (xmax, ymax), color=(255, 255, 0), thickness=2)
            cv2.polylines(tmp, [box[:, np.newaxis, :].astype(np.int)], isClosed=True, color=(255, 255, 0), thickness=2)
        vis_img = np.vstack([vis_img, tmp])
        cv2.imshow("image", vis_img)
        cv2.waitKey()
    return np.array(boxes, dtype=np.float32)
