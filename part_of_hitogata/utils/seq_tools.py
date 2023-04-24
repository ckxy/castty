import os
import math
import torch
import numpy as np
from collections import Iterable
from .utils import get_concat_h
from PIL import Image, ImageDraw, ImageFont
from nltk.metrics.distance import edit_distance


def draw_seq(img, text):
    if len(text) == 0:
        return img

    if not isinstance(img, Image.Image):
        is_np = True
        img = Image.fromarray(img)
    else:
        is_np = False

    w, h = img.size
    l = math.sqrt(h * h + w * w)
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)

    font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fonts', 'simsun.ttf')
    font = ImageFont.truetype(font_path, int(l * 5e-2))

    # text = seq2str(seq, chars)

    t_size = draw.textsize(text, font)
    draw.rectangle((0, 0, t_size[0], t_size[1]), fill=(0, 255, 255))
    draw.text((0, 0), text, fill=(0, 0, 0), font=font)

    if is_np:
        img = np.array(img)

    return img


def get_seq_from_batch(seq, lengths, index=0):
    inds = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(lengths, dim=0)], dim=0)
    return seq[inds[index]:inds[index + 1]].detach().cpu().numpy().astype(np.int)


def get_seqs_from_batch(seq, lengths):
    res = []
    inds = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(lengths, dim=0)], dim=0)
    for index in range(len(lengths)):
        res.append(seq[inds[index]:inds[index + 1]].detach().cpu().numpy().astype(np.int))
    return res


def seq2str(seq, chars):
    if not isinstance(seq[0], Iterable):
        res = ''
        for i in seq:
            res += chars[i]
        return res
    else:
        res = ['' for _ in seq]
        for k in range(len(seq)):
            for i in seq:
                res[k] += chars[i]
        return res


def logits2seqs(preds, chars, np_out=True):
    # greedy decode
    pred_labels = []
    labels = []
    for i in range(preds.shape[0]):
        pred = preds[i]
        pred_label = []
        for j in range(pred.shape[1]):
            pred_label.append(torch.argmax(pred[:, j], dim=0).item())
        no_repeat_blank_label = []
        pre_c = pred_label[0]
        for c in pred_label: # dropout repeate label and blank label
            if (pre_c == c) or (c == len(chars) - 1):
                if c == len(chars) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c

        if np_out:
            no_repeat_blank_label = np.array(no_repeat_blank_label)
        pred_labels.append(no_repeat_blank_label)
    
    # if np_out:
    #     np.array(pred_labels)
    return pred_labels


def calc_train_accuracy(pred_t, gt_t, lengths, chars):
    pred_seq = logits2seqs(pred_t, chars, True)
    gt_seq = get_seqs_from_batch(gt_t, lengths)
    return calc_accuracy(pred_seq, gt_seq)


def calc_accuracy(pred, gt):
    assert len(pred) == len(gt)
    total = len(pred)
    tp = 0
    for p, g in zip(pred, gt):
        # print(p, g)
        if p == g:
            tp += 1
    return tp / total


def calc_norm_ed(pred, gt):
    assert len(pred) == len(gt)
    norm_ED = 0

    for p, g in zip(pred, gt):
        if len(gt) == 0 or len(pred) == 0:
            norm_ED += 0
        elif len(gt) > len(pred):
            norm_ED += 1 - edit_distance(pred, gt) / len(gt)
        else:
            norm_ED += 1 - edit_distance(pred, gt) / len(pred)
    return norm_ED / len(pred)
