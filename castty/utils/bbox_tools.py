import os
import cv2
import math
import torch
import random
import colorsys
import numpy as np
from torchvision.ops import nms as tvnms
from itertools import product as product
from PIL import Image, ImageDraw, ImageFont


def xywh2xyxy(bboxes):
    if isinstance(bboxes, np.ndarray):
        bboxes = bboxes.copy()
    else:
        bboxes = bboxes.clone()
    bboxes[..., 0] -= 0.5 * bboxes[..., 2]
    bboxes[..., 1] -= 0.5 * bboxes[..., 3]
    bboxes[..., 2] += bboxes[..., 0]
    bboxes[..., 3] += bboxes[..., 1]
    return bboxes


def xyxy2xywh(bboxes):
    if isinstance(bboxes, np.ndarray):
        bboxes = bboxes.copy()
    else:
        bboxes = bboxes.clone()
    bboxes[..., 0] = 0.5 * (bboxes[..., 2] + bboxes[..., 0])
    bboxes[..., 1] = 0.5 * (bboxes[..., 3] + bboxes[..., 1])
    bboxes[..., 2] = 2 * (bboxes[..., 2] - bboxes[..., 0])
    bboxes[..., 3] = 2 * (bboxes[..., 3] - bboxes[..., 1])
    return bboxes


def grid_analysis(img, grid_sizes, bboxes_groups, indices_groups, bboxes_num, as_one=True):
    hsv_tuples = [(1.0 * x / bboxes_num, 1., 1.) for x in range(bboxes_num)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    if not isinstance(img, Image.Image):
        is_np = True
        img = Image.fromarray(img)
    else:
        is_np = False

    w, h = img.size
    assert len(grid_sizes) == len(bboxes_groups) == len(indices_groups)

    img_tmps = []
    for grid_size in grid_sizes:
        img_tmp = img.copy()
        draw = ImageDraw.Draw(img_tmp)
        for x in range(0, w - 1, grid_size):
            for y in range(0, h - 1, grid_size):
                draw.line((x, 0, x, h), width=1, fill=(0, 0, 0))
                draw.line((0, y, w, y), width=1, fill=(0, 0, 0))
        img_tmps.append(img_tmp)

    for i, indices in enumerate(indices_groups):
        draw = ImageDraw.Draw(img_tmps[i])
        for index in indices:
            x, y, b = index
            x, y, r = int((x + 0.5) * grid_sizes[i]), int((y + 0.5) * grid_sizes[i]), grid_sizes[i] // 4
            draw.ellipse((x - r, y - r, x + r, y + r), fill=colors[b])

    for i, bboxes in enumerate(bboxes_groups):
        draw = ImageDraw.Draw(img_tmps[i])
        for bbox in bboxes:
            x1, y1, x2, y2, b = bbox
            draw.rectangle((x1, y1, x2, y2), outline=colors[b], width=2)

    if as_one:
        s = np.array([img.size for img in img_tmps])
        # print(s, s.shape)
        dst = Image.new('RGB', (s[..., 0].sum() + (len(s) - 1) * 5, np.max(s[..., 1])))
        x = 0
        for i, img in enumerate(img_tmps):
            dst.paste(img, (x, 0))
            x += img.width + 5
        # dst.save('1.jpg')
        # exit()
        # return np.hstack(img_tmps[:-1]).astype(np.uint8)
        if is_np:
            dst = np.array(dst)
        return dst
    else:
        if is_np:
            dst = [np.array(img) for img in img_tmps]
        else:
            return [img for img in img_tmps]


def draw_bbox(img, bboxes, class_ids=None, classes=None, scores=None):
    if classes is not None:
        num_classes = len(classes)
        
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        # random.shuffle(colors)

    if not isinstance(img, Image.Image):
        is_np = True
        img = Image.fromarray(img)
    else:
        is_np = False

    w, h = img.size
    l = math.sqrt(h * h + w * w)

    draw = ImageDraw.Draw(img)

    font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fonts', 'arial.ttf')
    font = ImageFont.truetype(font_path, int(l * 1e-3 * 25))

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox, dtype=np.int32)

        if class_ids is None:
            draw.rectangle(tuple(coor), outline=(255, 0, 0), width=max(1, int(l / 600)))
        else:
            class_ind = int(class_ids[i])
            bbox_color = colors[class_ind]
            draw.rectangle(tuple(coor), outline=bbox_color, width=max(1, int(l / 600)))

            if scores is None:
                bbox_text = '{}'.format(classes[class_ind])
            else:
                bbox_text = '{}: {:.2f}'.format(classes[class_ind], scores[i])
            t_size = draw.textsize(bbox_text, font)
            text_box = (coor[0], coor[1] - t_size[1], coor[0] + t_size[0], coor[1])
            draw.rectangle(text_box, fill=bbox_color)
            draw.text(text_box[:2], bbox_text, fill=(0, 0, 0), font=font)

    if is_np:
        img = np.array(img)

    return img


def draw_bbox_without_label(img, bboxes):
    if not isinstance(img, Image.Image):
        is_np = True
        img = Image.fromarray(img)
    else:
        is_np = False

    w, h = img.size
    l = math.sqrt(h * h + w * w)

    draw = ImageDraw.Draw(img)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox, dtype=np.int32)
        draw.rectangle(tuple(coor), outline=(255, 0, 0), width=max(1, int(l / 600)))

    if is_np:
        img = np.array(img)

    return img


def calc_diou(boxes1, boxes2):
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制，且需要是Tensor
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(xmin, ymin, xmax, ymax)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1 = torch.cat([torch.min(boxes1[..., :2], boxes1[..., 2:]),
                        torch.max(boxes1[..., :2], boxes1[..., 2:])], dim=-1)
    boxes2 = torch.cat([torch.min(boxes2[..., :2], boxes2[..., 2:]),
                        torch.max(boxes2[..., :2], boxes2[..., 2:])], dim=-1)

    center_x1 = (boxes1[..., 2] + boxes1[..., 0]) / 2
    center_y1 = (boxes1[..., 3] + boxes1[..., 1]) / 2
    center_x2 = (boxes2[..., 2] + boxes2[..., 0]) / 2
    center_y2 = (boxes2[..., 3] + boxes2[..., 1]) / 2
    # w1 = boxes1[..., 2] - boxes1[..., 0]
    # h1 = boxes1[..., 3] - boxes1[..., 1]
    # w2 = boxes2[..., 2] - boxes2[..., 0]
    # h2 = boxes2[..., 3] - boxes2[..., 1]

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
    intersection_left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    intersection_right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    intersection = torch.max(intersection_right_down - intersection_left_up, torch.zeros_like(intersection_right_down))
    inter_area = intersection[..., 0] * intersection[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area

    enclose_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    enclose = torch.max(enclose_right_down - enclose_left_up, torch.zeros_like(enclose_left_up))
    outer_diag = enclose[..., 0] ** 2 + enclose[..., 1] ** 2
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2

    DIOU = IOU - (1.0 * inter_diag / outer_diag)
    DIOU = torch.clamp(DIOU, min=-1.0, max=1.0)
    return DIOU


def calc_iou1(boxes1, boxes2):
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(xmin, ymin, xmax, ymax)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 计算出boxes1和boxes2相交部分的左上角坐标、右下角坐标
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # 计算出boxes1和boxes2相交部分的宽、高
    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU


def calc_iou2(boxes1, boxes2, iou_loss=False):
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制，且需要是Tensor
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(xmin, ymin, xmax, ymax)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
    left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = torch.max(right_down - left_up, torch.zeros_like(right_down))
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    if not iou_loss:
        IOU = inter_area / union_area
    else:
        IOU = (inter_area + 1) / (union_area + 1)
    return IOU


def calc_giou(boxes1, boxes2):
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制，且需要是Tensor
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(xmin, ymin, xmax, ymax)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """

    boxes1 = torch.cat([torch.min(boxes1[..., :2], boxes1[..., 2:]),
                        torch.max(boxes1[..., :2], boxes1[..., 2:])], dim=-1)
    boxes2 = torch.cat([torch.min(boxes2[..., :2], boxes2[..., 2:]),
                        torch.max(boxes2[..., :2], boxes2[..., 2:])], dim=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
    intersection_left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    intersection_right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    intersection = torch.max(intersection_right_down - intersection_left_up, torch.zeros_like(intersection_right_down))
    inter_area = intersection[..., 0] * intersection[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area

    enclose_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    enclose = torch.max(enclose_right_down - enclose_left_up, torch.zeros_like(enclose_left_up))
    enclose_area = enclose[..., 0] * enclose[..., 1]
    GIOU = IOU - 1.0 * (enclose_area - union_area) / enclose_area

    return GIOU


def nms_per_class(boxes, scores, overlap=0.45, top_k=200, variance=None, vvsigma=0.05):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = torch.Tensor(scores.size(0)).fill_(0).long()
    if boxes.numel() == 0:
        return keep

    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    # idx = idx[-top_k:]  # indices of the top-k largest vals

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        iou = calc_iou2(boxes[i].unsqueeze(0), boxes[idx])

        if variance is not None:
            iou_mask = iou > 0
            kl_box = boxes[idx][iou_mask]
            kl_box = torch.cat((kl_box, boxes[i].unsqueeze(0)), 0)
            kl_iou = iou[iou_mask]
            kl_var = torch.cat((variance[idx][iou_mask], variance[i].unsqueeze(0)), 0)
            p = torch.exp(-1 * torch.pow((1 - kl_iou), 2) / vvsigma)
            p = torch.cat((p, torch.ones(1).to(boxes.device)), 0).unsqueeze(1)
            p = p / kl_var
            p = p / p.sum(0)
            boxes[i] = (p * kl_box[:, :4]).sum(0)

        idx = idx[iou.le(overlap)]

    keep = keep[:count]

    return boxes[keep], scores[keep], keep


def nms(boxes, scores, iou_thresh=0.45, score_thresh=0.05, **kwargs):
    if 'variance' in kwargs.keys():
        vari = kwargs['variance']
        vvsigma = kwargs['vvsigma']
    else:
        vari = None
        vvsigma = 0

    leave = False
    if 'indices' in kwargs.keys():
        if kwargs['indices']:
            leave = True

    res = []
    for j in range(scores.shape[1]):
        inds = torch.where(scores[:, j] > score_thresh)[0]
        if len(inds) == 0:
            continue

        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        if vari is not None:
            c_vari = vari[inds].clone()
        else:
            c_vari = None
        # print(j, c_bboxes.shape)

        c_bboxes, c_scores, keep = nms_per_class(c_bboxes.clone(), c_scores.clone(), iou_thresh, variance=c_vari, vvsigma=vvsigma)

        c_scores = c_scores.unsqueeze(-1)
        c_label = torch.ones(c_scores.shape).to(c_scores.device) * j
        if leave:
            bbox = torch.cat([inds[keep].unsqueeze(-1).type(c_bboxes.dtype), c_bboxes, c_scores, c_label], dim=-1)
        else:
            bbox = torch.cat([c_bboxes, c_scores, c_label], dim=-1)
        res.append(bbox)

    # exit()

    if len(res) > 0:
        return torch.cat(res, dim=0)
    else:
        return torch.zeros(0, 6)


def get_ssd_priors(**kwargs):
    mean = []
    for k, f in enumerate(kwargs['shapes']):
        for i, j in product(range(f), repeat=2):
            f_k = kwargs['image_size'] / kwargs['steps'][k]
            cx = (j + 0.5) / f_k
            cy = (i + 0.5) / f_k

            s_k = kwargs['min_sizes'][k] / kwargs['image_size']
            mean += [cx, cy, s_k, s_k]

            # aspect_ratio: 1
            # rel size: sqrt(s_k * s_(k+1))
            s_k_prime = math.sqrt(s_k * (kwargs['max_sizes'][k] / kwargs['image_size']))
            mean += [cx, cy, s_k_prime, s_k_prime]

            # rest of aspect ratios
            if isinstance(kwargs['aspect_ratios'][k], int):
                ar = kwargs['aspect_ratios'][k]
                mean += [cx, cy, s_k * math.sqrt(ar), s_k / math.sqrt(ar)]
                mean += [cx, cy, s_k / math.sqrt(ar), s_k * math.sqrt(ar)]
            else:
                for ar in kwargs['aspect_ratios'][k]:
                    mean += [cx, cy, s_k * math.sqrt(ar), s_k / math.sqrt(ar)]
                    mean += [cx, cy, s_k / math.sqrt(ar), s_k * math.sqrt(ar)]

    priors = np.array(mean).reshape(-1, 4).astype(np.float32)
    if kwargs['clip']:
        priors = np.clip(priors, 0, 1)

    return priors


def ori_ssd_encode1(coor, priors, variances=(1, 1)):
    res = coor.copy()
    res[..., :2] -= priors[..., :2]
    res[..., :2] /= variances[0] * priors[..., 2:]
    res[..., 2:] /= priors[..., 2:]
    res[..., 2:] = np.log(res[..., 2:]) / variances[1]
    return res


def ori_ssd_encode2(coor, priors, variances=(1, 1)):
    res = coor.clone()
    res[..., :2] -= priors[..., :2]
    res[..., :2] /= variances[0] * priors[..., 2:]
    res[..., 2:] /= priors[..., 2:]
    res[..., 2:] = torch.log(res[..., 2:]) / variances[1]
    return res


def ori_ssd_decode1(loc, priors, variances=(1, 1)):
    res = loc.copy()
    res[..., :2] = priors[..., :2] + loc[..., :2] * variances[0] * priors[..., 2:]
    res[..., 2:] = priors[..., 2:] * np.exp(loc[..., 2:] * variances[1])
    return res


def ori_ssd_decode2(loc, priors, variances=(1, 1)):
    res = loc.clone()
    res[..., :2] = priors[..., :2] + loc[..., :2] * variances[0] * priors[..., 2:]
    res[..., 2:] = priors[..., 2:] * torch.exp(loc[..., 2:] * variances[1])
    return res


def get_xy_grid1(h, w, gt_per_grid):
    shiftx = np.arange(w)
    shifty = np.arange(h)
    shiftx, shifty = np.meshgrid(shiftx, shifty)
    shiftx = np.repeat(shiftx[..., np.newaxis], gt_per_grid, axis=-1)[..., np.newaxis]
    shifty = np.repeat(shifty[..., np.newaxis], gt_per_grid, axis=-1)[..., np.newaxis]
    xy_grid = np.concatenate([shiftx, shifty], axis=-1)
    return xy_grid


def get_xy_grid2(h, w, gt_per_grid):
    shiftx = torch.arange(0, w, dtype=torch.float32)
    shifty = torch.arange(0, h, dtype=torch.float32)
    shiftx, shifty = torch.meshgrid([shifty, shiftx], indexing='xy')
    shiftx = shiftx.unsqueeze(-1).repeat(1, 1, gt_per_grid)
    shifty = shifty.unsqueeze(-1).repeat(1, 1, gt_per_grid)
    xy_grid = torch.stack([shiftx, shifty], dim=-1)
    # print(xy_grid)
    # exit()
    return xy_grid


def exyxy_decode2(loc, grids):
    x1y1 = grids + 0.5 - torch.exp(loc[..., :2])
    x2y2 = grids + 0.5 + torch.exp(loc[..., 2:])
    xyxy = torch.cat((x1y1, x2y2), dim=-1)
    return xyxy


def sxyewh_decode2(loc, grids):
    xy = grids + torch.sigmoid(loc[..., :2])
    wh = torch.exp(loc[..., 2:])
    xywh = torch.cat((xy, wh), dim=-1)
    xyxy = xywh2xyxy(xywh)
    return xyxy


def get_grid1(h, w, center=True):
    if center:
        c = 0.5
    else:
        c = 1
    shiftx = np.arange(w) + c
    shifty = np.arange(h) + c
    shiftx, shifty = np.meshgrid(shiftx, shifty)
    return shiftx, shifty


def get_grid2(h, w, center=True):
    if center:
        c = 0.5
    else:
        c = 1
    shiftx = torch.arange(0, w, dtype=torch.float32) + c
    shifty = torch.arange(0, h, dtype=torch.float32) + c
    shifty, shiftx = torch.meshgrid([shifty, shiftx], indexing='ij')
    return shifty, shiftx


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        bbox (Tensor): Shape (n, 4), "xyxy" format
        max_dis (float): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=

    Returns:
        Tensor: Decoded distances.
    """
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # (TODO): as ONNX does not support repeat now,
    # we have to use this ugly code
    bboxes = torch.masked_select(
        bboxes,
        torch.stack((valid_mask, valid_mask, valid_mask, valid_mask),
                    -1)).view(-1, 4)
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        return bboxes, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]


def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.
    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.
    Arguments:
        boxes (torch.Tensor): boxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict): specify nms type and other parameters like iou_thr.
            Possible keys includes the following.
            - iou_thr (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
                number of boxes is large (e.g., 200k). To avoid OOM during
                training, the users could set `split_thr` to a small value.
                If the number of boxes is greater than the threshold, it will
                perform NMS on each group of boxes separately and sequentially.
                Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class.
    Returns:
        tuple: kept dets and indice.
    """
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]

    nms_type = nms_cfg_.pop('type', 'nms')
    # nms_op = eval(nms_type)

    split_thr = nms_cfg_.pop('split_thr', 10000)
    if len(boxes_for_nms) < split_thr:
        # dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        keep = tvnms(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]
        # scores = dets[:, -1]
        scores = scores[keep]
    else:
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            # dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            keep = tvnms(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True

        keep = total_mask.nonzero(as_tuple=False).view(-1)
        keep = keep[scores[keep].argsort(descending=True)]
        boxes = boxes[keep]
        scores = scores[keep]

    return torch.cat([boxes, scores[:, None]], -1), keep
