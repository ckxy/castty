import os
import cv2
import math
import torch
import numpy as np
from .utils import get_concat_h
from PIL import Image, ImageDraw, ImageFont


def draw_label(img, labels, classes, scores=None):
    if len(classes) > 1:
        return draw_grouped_label(img, labels, classes, scores)

    if isinstance(labels, int) or isinstance(labels, float):
        labs = [labels]
    else:
        if isinstance(labels, np.ndarray) and labels.shape == ():
            labs = [labels]
        else:
            labs = labels

    if scores is not None:
        if isinstance(scores, int) or isinstance(scores, float):
            scs = [scores]
        else:
            if isinstance(scores, np.ndarray) and scores.shape == ():
                scs = [scores]
            else:
                scs = scores

        assert len(labs) == len(scs)
    else:
        scs = None

    text = ''
    for i, label in enumerate(labs):
        text += '{}'.format(classes[0][label])
        if scs is not None:
            text += ': {:.3f}'.format(scs[i])
        text += '\n'
    text = text[:-1]

    if not isinstance(img, Image.Image):
        is_np = True
        img = Image.fromarray(img)
    else:
        is_np = False

    w, h = img.size
    l = math.sqrt(h * h + w * w)
    draw = ImageDraw.Draw(img)
    font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fonts', 'arial.ttf')
    font = ImageFont.truetype(font_path, int(l * 5e-2))

    t_size = draw.multiline_textsize(text, font)
    draw.rectangle((0, 0, t_size[0], t_size[1]), fill=(0, 255, 255))
    draw.multiline_text((0, 0), text, fill=(0, 0, 0), font=font)

    if is_np:
        img = np.asarray(img)

    return img


def draw_grouped_label(img, labels, classes, scores=None):
    if scores is not None:
        assert len(labels) == len(scores)

    w, h = img.size
    l = math.sqrt(h * h + w * w)
    font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fonts', 'arial.ttf')
    font = ImageFont.truetype(font_path, int(l * 3e-2))

    bar = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(bar)

    text = ''
    for i, label in enumerate(labels):
        if len(classes[i]) == 1:
            if scores is None:
                text += '{}: {:.3f}'.format(classes[i][0], label)
            else:
                text += '{}: {:.3f}'.format(classes[i][0], scores[i])
        else:
            text += '{}'.format(classes[i][label])
            if scores is not None and scores[i] != -1:
                text += ': {:.3f}'.format(scores[i])
        text += '\n'
    text = text[:-1]

    draw.multiline_text((0, 0), text, fill=(0, 0, 0), font=font)

    img = get_concat_h(img, bar)

    return img


def calc_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


def calc_mean_accuracy(pred, target):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        batch_size, num_groups = target.shape

        correct = pred.eq(target)
        correct = correct.view(-1).float().sum(0)
        macc = correct.mul_(100.0 / (batch_size * num_groups)).item()

    return macc


def logits_decode(output, topk=(1,), score=False):
    res = []

    if score:
        scores = torch.nn.functional.softmax(output)
    else:
        scores = output.clone()

    maxk = max(topk)
    if output.dim() == 1:
        value, ind = scores.unsqueeze(0).topk(maxk, 1, True, True)
        value = value.squeeze(0)
        ind = ind.squeeze(0)

        for k in topk:
            if score:
                res.append([ind[:k], value[:k]])
            else:
                res.append([ind[:k], None])
    else:
        value, ind = scores.topk(maxk, 1, True, True)

        for k in topk:
            if score:
                res.append([ind[:, :k], value[:, :k]])
            else:
                res.append([ind[:, :k], None])

    return res
