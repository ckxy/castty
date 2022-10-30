import cv2
import math
import torch
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


def gaussian2D(radius, sigma=1, dtype=torch.float32, device='cpu'):
    """Generate 2D gaussian kernel.

    Args:
        radius (int): Radius of gaussian kernel.
        sigma (int): Sigma of gaussian function. Default: 1.
        dtype (torch.dtype): Dtype of gaussian tensor. Default: torch.float32.
        device (str): Device of gaussian tensor. Default: 'cpu'.

    Returns:
        h (Tensor): Gaussian kernel with a
            ``(2 * radius + 1) * (2 * radius + 1)`` shape.
    """
    x = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device).view(1, -1)
    y = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device).view(-1, 1)

    h = (-(x * x + y * y) / (2 * sigma * sigma)).exp()

    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h


def gen_gaussian_target(heatmap, center, radius, k=1):
    """Generate 2D gaussian heatmap.

    Args:
        heatmap (Tensor): Input heatmap, the gaussian kernel will cover on
            it and maintain the max value.
        center (list[int]): Coord of gaussian kernel's center.
        radius (int): Radius of gaussian kernel.
        k (int): Coefficient of gaussian kernel. Default: 1.

    Returns:
        out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
    """
    diameter = 2 * radius + 1
    gaussian_kernel = gaussian2D(
        radius, sigma=diameter / 6, dtype=heatmap.dtype, device=heatmap.device)

    x, y = center

    height, width = heatmap.shape[:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian_kernel[radius - top:radius + bottom,
                                      radius - left:radius + right]
    out_heatmap = heatmap
    torch.max(
        masked_heatmap,
        masked_gaussian * k,
        out=out_heatmap[y - top:y + bottom, x - left:x + right])

    return out_heatmap


def gaussian_radius(det_size, min_overlap):
    """
    https://github.com/princeton-vl/CornerNet/issues/110
    https://blog.csdn.net/weixin_41722370/article/details/119655256
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)


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
