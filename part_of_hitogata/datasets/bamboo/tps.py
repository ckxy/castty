import os
import torch
import numpy as np
from PIL import Image
from .builder import INTERNODE
import torch.nn.functional as F
from .base_internode import BaseInternode
from ..utils.common import get_image_size, is_pil
from torchvision.transforms.functional import to_tensor, to_pil_image


__all__ = ['TPSStretch', 'TPSDistort']


def calc_points(length):
    a1, a2 = -1, 1
    l = length - 1

    res = []
    for i in range(length):
        res.append(((l - i) * a1 + i * a2) / l)
    return res


def build_P(width, height):
    x = calc_points(width)
    y = calc_points(height)

    shiftx, shifty = np.meshgrid(x, y)
    xy = np.stack([shiftx, shifty], axis=-1).astype(np.float32).reshape(-1, 2)
    return xy  # n (= self.I_r_width x self.I_r_height) x 2


def build_inv_delta_C(F, C):
    """ Return inv_delta_C which is needed to calculate T """
    hat_C = np.zeros((F, F), dtype=float)  # F x F
    for i in range(0, F):
        for j in range(i, F):
            r = np.linalg.norm(C[i] - C[j])
            hat_C[i, j] = r
            hat_C[j, i] = r
    np.fill_diagonal(hat_C, 1)
    hat_C = (hat_C ** 2) * np.log(hat_C)
    # print(C.shape, hat_C.shape)
    delta_C = np.concatenate(  # F+3 x F+3
        [
            np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),  # F x F+3
            np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),  # 2 x F+3
            np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1)  # 1 x F+3
        ],
        axis=0
    )
    inv_delta_C = np.linalg.inv(delta_C)
    return inv_delta_C  # F+3 x F+3


def build_P_hat(F, C, P, eps=1e-6):
    n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
    P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))  # n x 2 -> n x 1 x 2 -> n x F x 2
    C_tile = np.expand_dims(C, axis=0)  # 1 x F x 2
    P_diff = P_tile - C_tile  # n x F x 2
    rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)  # n x F
    rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + eps))  # n x F
    P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
    return P_hat  # n x F+3


def build_P_prime(tgt_cps, inv_delta_C, P_hat):
    tgt_cps_with_zeros = np.concatenate([tgt_cps, np.zeros([3, 2], dtype=np.float32)], axis=0)
    T = inv_delta_C @ tgt_cps_with_zeros  # F+3 x 2
    grids = P_hat @ T  # n x 2
    return grids.astype(np.float32)  # n x 2


def gen_grid(size, src_cps, tgt_cps, eps=1e-6, resize=False):
    width, height = size

    if resize:
        x1 = np.min(src_cps[..., 0])
        x2 = np.max(src_cps[..., 0]) 
        y1 = np.min(src_cps[..., 1])
        y2 = np.max(src_cps[..., 1])
        r = min(2 / (x2 - x1), 2 / (y2 - y1))

        resized_src_cps = r * src_cps

        x1 = np.min(resized_src_cps[..., 0])
        y1 = np.min(resized_src_cps[..., 1])
        
        resized_src_cps[..., 0] -= x1 + 1
        resized_src_cps[..., 1] -= y1 + 1
    else:
        resized_src_cps = src_cps
    
    n_cps = len(resized_src_cps)

    P = build_P(width, height)  # 全图控制点，范围-1，1

    inv_delta_C = build_inv_delta_C(n_cps, resized_src_cps)  # F+3 x F+3
    P_hat = build_P_hat(n_cps, resized_src_cps, P, eps=1e-6)  # n x F+3
    grids = build_P_prime(tgt_cps, inv_delta_C, P_hat)
    grids = grids.reshape(height, width, 2)
    return grids


class TPS(BaseInternode):
    def __init__(self, segment, resize=False, **kwargs):
        assert segment > 1

        self.segment = segment
        self.resize = resize

    def calc_intl_param_forward(self, data_dict):
        raise NotImplementedError

    def forward(self, data_dict):
        if not is_pil(data_dict['image']):
            data_dict['image'] = Image.fromarray(data_dict['image'])
            is_np = True
        else:
            is_np = False

        grids = gen_grid(get_image_size(data_dict['image']), data_dict['intl_tps_src_cps'], data_dict['intl_tps_tgt_cps'], resize=self.resize)

        img = to_tensor(data_dict['image'])
        img = img.unsqueeze(0)
        img = F.grid_sample(img, torch.from_numpy(grids).unsqueeze(0), align_corners=True)
        data_dict['image'] = to_pil_image(img[0])
        
        if is_np:
            data_dict['image'] = np.asarray(data_dict['image'])
        return data_dict

    def erase_intl_param_forward(self, data_dict):
        data_dict.pop('intl_tps_src_cps')
        data_dict.pop('intl_tps_tgt_cps')
        return data_dict

    def __repr__(self):
        raise NotImplementedError


@INTERNODE.register_module()
class TPSStretch(TPS):
    def __init__(self, segment, **kwargs):
        super(TPSStretch, self).__init__(segment, False)

    @staticmethod
    def stretch(size, segment):
        assert segment > 1
        img_w, img_h = size

        cut = img_w // segment
        thresh = cut * 4 // 5

        src_pts = list()
        dst_pts = list()

        src_pts.append([0, 0])
        src_pts.append([img_w, 0])
        src_pts.append([img_w, img_h])
        src_pts.append([0, img_h])

        dst_pts.append([0, 0])
        dst_pts.append([img_w, 0])
        dst_pts.append([img_w, img_h])
        dst_pts.append([0, img_h])

        half_thresh = thresh * 0.5

        for cut_idx in np.arange(1, segment, 1):
            move = np.random.randint(thresh) - half_thresh
            src_pts.append([cut * cut_idx, 0])
            src_pts.append([cut * cut_idx, img_h])
            dst_pts.append([cut * cut_idx + move, 0])
            dst_pts.append([cut * cut_idx + move, img_h])

        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)

        src_pts[..., 0] /= img_w
        src_pts[..., 1] /= img_h
        src_pts = (src_pts - 0.5) * 2

        dst_pts[..., 0] /= img_w
        dst_pts[..., 1] /= img_h
        dst_pts = (dst_pts - 0.5) * 2

        return src_pts, dst_pts

    def calc_intl_param_forward(self, data_dict):
        data_dict['intl_tps_tgt_cps'], data_dict['intl_tps_src_cps'] = self.stretch(get_image_size(data_dict['image']), self.segment)
        return data_dict

    def __repr__(self):
        return 'TPSStretch(segment={})'.format(self.segment)


@INTERNODE.register_module()
class TPSDistort(TPS):
    @staticmethod
    def distort(size, segment):
        assert segment > 1
        img_w, img_h = size

        cut = img_w // segment

        src_pts = list()
        dst_pts = list()

        src_pts.append([0, 0])
        src_pts.append([img_w, 0])
        src_pts.append([img_w, img_h])
        src_pts.append([0, img_h])

        for cut_idx in np.arange(1, segment, 1):
            src_pts.append([cut * cut_idx, 0])
            src_pts.append([cut * cut_idx, img_h])

        src_pts = np.array(src_pts, dtype=np.float32)

        src_pts[..., 0] /= img_w
        src_pts[..., 1] /= img_h
        src_pts = (src_pts - 0.5) * 2

        dst_pts = src_pts.copy()

        dst_pts[..., 0] += np.random.uniform(-0.5 / segment, 0.5 / segment, dst_pts[..., 0].shape)
        dst_pts[..., 1] += np.random.uniform(-1 / segment, 1 / segment, dst_pts[..., 1].shape)

        dst_pts = np.array(dst_pts, dtype=np.float32)

        return src_pts, dst_pts

    def calc_intl_param_forward(self, data_dict):
        data_dict['intl_tps_tgt_cps'], data_dict['intl_tps_src_cps'] = self.distort(get_image_size(data_dict['image']), self.segment)
        return data_dict

    def __repr__(self):
        return 'TPSDistort(segment={}, resize={})'.format(self.segment, self.resize)
