import math
import random
import numpy as np
from .bamboo import Bamboo
from .builder import INTERNODE
from .builder import build_internode
from .warp_internode import WarpInternode
from ..utils.warp_tools import calc_expand_size_and_matrix, warp_bbox, warp_point
from ..utils.common import get_image_size, clip_bbox, filter_bbox


__all__ = ['Warp', 'WarpPerspective', 'WarpResize', 'WarpScale', 'WarpStretch', 'WarpRotate', 'WarpShear', 'WarpTranslate']


@INTERNODE.register_module()
class Warp(Bamboo):
    def __init__(self, internodes, ccs=True, **kwargs):
        assert len(internodes) > 0

        self.internode = WarpInternode(ccs=ccs)
        self.internodes = []
        self.ccs = ccs

        for cfg in internodes:
            cfg['ccs'] = False
            self.internodes.append(build_internode(cfg))

    def __call__(self, data_dict):
        data_dict['warp_matrix'] = np.eye(3)
        data_dict['warp_size'] = get_image_size(data_dict['image'])

        data_dict = super(Warp, self).__call__(data_dict)

        data_dict['warp_tmp_matrix'] = data_dict.pop('warp_matrix')
        data_dict['warp_tmp_size'] = data_dict.pop('warp_size')

        data_dict = self.internode(data_dict)

        return data_dict

    def reverse(self, **kwargs):
        return kwargs

    def __repr__(self):
        split_str = [i.__repr__() for i in self.internodes]
        bamboo_str = type(self).__name__ + '('
        for i in range(len(split_str)):
            bamboo_str += '\n  ' + split_str[i].replace('\n', '\n  ')
        bamboo_str += '\n  ccs={}'.format(self.ccs)
        bamboo_str = '{}\n)'.format(bamboo_str)

        return bamboo_str

    def rper(self):
        return 'Warp(not available)'


@INTERNODE.register_module()
class WarpPerspective(WarpInternode):
    def __init__(self, distortion_scale=0.5, **kwargs):
        super(WarpPerspective, self).__init__(**kwargs)
        self.distortion_scale = distortion_scale

    @staticmethod
    def get_params(width, height, distortion_scale):
        """Get parameters for ``perspective`` for a random perspective transform.

        Args:
            width : width of the image.
            height : height of the image.

        Returns:
            List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
            List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
        """
        half_height = int(height / 2)
        half_width = int(width / 2)
        topleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(0, int(distortion_scale * half_height)))
        topright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(0, int(distortion_scale * half_height)))
        botright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        botleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        endpoints = [topleft, topright, botright, botleft]
        # endpoints = [(83, 18), (605, 25), (605, 397), (139, 341)]
        return startpoints, endpoints

    @staticmethod
    def build_matrix(startpoints, endpoints):
        matrix = []
        for p1, p2 in zip(startpoints, endpoints):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        A = np.array(matrix)
        B = np.array(endpoints).flatten()
        c, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        c = c.tolist() + [1]
        c = np.matrix(c).reshape(3, 3)
        return np.array(c)

    def __call__(self, data_dict):
        if 'warp_size' in data_dict.keys():
            size = data_dict['warp_size']
        else:
            size = get_image_size(data_dict['image'])

        width, height = size
        startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
        M = self.build_matrix(startpoints, endpoints)

        if self.expand:
            # print(startpoints, endpoints)
            
            xx = [e[0] for e in endpoints]
            yy = [e[1] for e in endpoints]
            nw = max(xx) - min(xx)
            nh = max(yy) - min(yy)
            E = np.eye(3)
            E[0, 2] = -min(xx)
            E[1, 2] = -min(yy)

            M = E @ M
            size = (nw, nh)
            # print(size, 'new')

        data_dict['warp_tmp_matrix'] = M
        data_dict['warp_tmp_size'] = size
        data_dict = super(WarpPerspective, self).__call__(data_dict)

        return data_dict

    def __repr__(self):
        return 'WarpPerspective(distortion_scale={}, {})'.format(self.distortion_scale, super(WarpPerspective, self).__repr__())


@INTERNODE.register_module()
class WarpResize(WarpInternode):
    def __init__(self, size, keep_ratio=True, short=False, **kwargs):
        super(WarpResize, self).__init__(**kwargs)

        assert len(size) == 2
        assert size[0] > 0 and size[1] > 0
        self.size = size
        self.keep_ratio = keep_ratio
        self.short = short

    def build_matrix(self, img_size):
        w, h = img_size

        C = np.eye(3)
        C[0, 2] = -w / 2
        C[1, 2] = -h / 2

        R = np.eye(3)
        if self.keep_ratio:
            if self.short:
                r = max(self.size[0] / w, self.size[1] / h)
            else:
                r = min(self.size[0] / w, self.size[1] / h)
            R[0, 0] = r
            R[1, 1] = r

            ow = (self.size[0] - R[0, 0] * w) / 2
            oh = (self.size[1] - R[1, 1] * h) / 2
        else:
            R[0, 0] = self.size[0] / w
            R[1, 1] = self.size[1] / h

            ow = 0
            oh = 0

        CI = np.eye(3)

        # if self.center:
        #     CI[0, 2] = self.size[0] / 2
        #     CI[1, 2] = self.size[1] / 2
        # else:
        CI[0, 2] = self.size[0] / 2 - ow
        CI[1, 2] = self.size[1] / 2 - oh

        return CI @ R @ C

    def calc_scale(self, size):
        w, h = size
        tw, th = self.size
        rw, rh = tw / w, th / h

        if self.keep_ratio:
            if self.short:
                r = max(rh, rw)
                scale = (r, r)
            else:
                r = min(rh, rw)
                scale = (r, r)
        else:
            scale = (rw, rh)
        return scale

    def __call__(self, data_dict):
        if 'warp_size' in data_dict.keys():
            size = data_dict['warp_size']
        else:
            size = get_image_size(data_dict['image'])

        M = self.build_matrix(size)

        if self.keep_ratio and (self.expand or self.short):
            _, new_size = calc_expand_size_and_matrix(M, size)
            size = new_size
        else:
            size = self.size

        # print(M, 'M1')

        data_dict['warp_tmp_matrix'] = M
        data_dict['warp_tmp_size'] = size
        data_dict = super(WarpResize, self).__call__(data_dict)

        return data_dict

    def reverse(self, **kwargs):
        if 'resize_and_padding_reverse_flag' not in kwargs.keys():
            return kwargs

        if 'ori_size' in kwargs.keys():
            h, w = kwargs['ori_size']
            h, w = int(h), int(w)
            M = self.build_matrix((w, h))
            # print(M, 'M2', type(M))
            # print(np.matrix(M).I)
            M = np.array(np.matrix(M).I)
        else:
            return kwargs

        if 'bbox' in kwargs.keys():
            wargs['bbox'] = warp_bbox(kwargs['bbox'], M)
            # boxes = clip_bbox(boxes, (w, h))
            # keep = filter_bbox(boxes)
            # kwargs['bbox'] = boxes[keep]

            # if 'bbox_meta' in kwargs.keys():
            #     kwargs['bbox_meta'].filter(keep)

        if 'poly' in kwargs.keys():
            # print(kwargs['poly'])
            kwargs['poly'] = [warp_point(p, M) for p in kwargs['poly']]
            # kwargs['poly'], keep = clip_poly(kwargs['poly'], dst_size)
            
            # if 'poly_meta' in kwargs.keys():
            #     kwargs['poly_meta'].filter(keep)

        return kwargs

    def __repr__(self):
        return 'WarpResize(size={}, keep_ratio={}, short={}, {})'.format(self.size, self.keep_ratio, self.short, super(WarpResize, self).__repr__())


@INTERNODE.register_module()
class WarpScale(WarpInternode):
    def __init__(self, r, **kwargs):
        super(WarpScale, self).__init__(**kwargs)

        assert len(r) == 2
        assert r[0] <= r[1] and r[0] > 0
        self.r = tuple(r)

    @staticmethod
    def build_matrix(r, img_size):
        w, h = img_size

        C = np.eye(3)
        C[0, 2] = -w / 2
        C[1, 2] = -h / 2

        R = np.eye(3)
        R[0, 0] = r
        R[1, 1] = r

        CI = np.eye(3)
        CI[0, 2] = w / 2
        CI[1, 2] = h / 2

        return CI @ R @ C

    def __call__(self, data_dict):
        if 'warp_size' in data_dict.keys():
            size = data_dict['warp_size']
        else:
            size = get_image_size(data_dict['image'])

        r = random.uniform(*self.r)
        M = self.build_matrix(r, size)

        data_dict['warp_tmp_matrix'] = M
        data_dict['warp_tmp_size'] = size
        data_dict = super(WarpScale, self).__call__(data_dict)

        return data_dict

    def __repr__(self):
        return 'WarpScale(r={}, {})'.format(self.r, super(WarpScale, self).__repr__())


@INTERNODE.register_module()
class WarpStretch(WarpInternode):
    def __init__(self, rw, rh, **kwargs):
        super(WarpStretch, self).__init__(**kwargs)

        assert len(rw) == 2 and len(rh) == 2
        assert rw[0] <= rw[1] and rw[0] > 0
        assert rh[0] <= rh[1] and rh[0] > 0
        self.rw = tuple(rw)
        self.rh = tuple(rh)

    @staticmethod
    def build_matrix(rs, img_size):
        w, h = img_size

        C = np.eye(3)
        C[0, 2] = -w / 2
        C[1, 2] = -h / 2

        R = np.eye(3)
        R[0, 0] = rs[0]
        R[1, 1] = rs[1]

        CI = np.eye(3)
        CI[0, 2] = w / 2
        CI[1, 2] = h / 2

        return CI @ R @ C

    def __call__(self, data_dict):
        if 'warp_size' in data_dict.keys():
            size = data_dict['warp_size']
        else:
            size = get_image_size(data_dict['image'])

        rw = random.uniform(*self.rw)
        rh = random.uniform(*self.rh)
        M = self.build_matrix((rw, rh), size)

        data_dict['warp_tmp_matrix'] = M
        data_dict['warp_tmp_size'] = size
        data_dict = super(WarpStretch, self).__call__(data_dict)

        return data_dict

    def __repr__(self):
        return 'WarpStretch(rw={}, rh={}, {})'.format(self.rw, self.rh, super(WarpStretch, self).__repr__())


@INTERNODE.register_module()
class WarpRotate(WarpInternode):
    def __init__(self, angle, **kwargs):
        super(WarpRotate, self).__init__(**kwargs)

        assert -180 < angle[0] <= 180
        assert -180 < angle[1] <= 180
        assert angle[0] <= angle[1]
        self.angle = angle

    @staticmethod
    def build_matrix(angle, img_size):
        w, h = img_size
        angle = math.radians(angle)

        C = np.eye(3)
        C[0, 2] = -w / 2
        C[1, 2] = -h / 2

        R = np.eye(3)
        R[0, 0] = round(math.cos(angle), 15)
        R[0, 1] = -round(math.sin(angle), 15)
        R[1, 0] = round(math.sin(angle), 15)
        R[1, 1] = round(math.cos(angle), 15)

        CI = np.eye(3)
        CI[0, 2] = w / 2
        CI[1, 2] = h / 2

        return CI @ R @ C

    def __call__(self, data_dict):
        angle = random.uniform(self.angle[0], self.angle[1])

        if angle != 0:
            if 'warp_size' in data_dict.keys():
                size = data_dict['warp_size']
            else:
                size = get_image_size(data_dict['image'])

            M = self.build_matrix(angle, size)

            if self.expand:
                E, new_size = calc_expand_size_and_matrix(M, size)
                M = E @ M
                size = new_size

            data_dict['warp_tmp_matrix'] = M
            data_dict['warp_tmp_size'] = size
            data_dict = super(WarpRotate, self).__call__(data_dict)

        return data_dict

    def __repr__(self):
        return 'WarpRotate(angle={}, {})'.format(self.angle, super(WarpRotate, self).__repr__())


@INTERNODE.register_module()
class WarpShear(WarpInternode):
    def __init__(self, ax, ay, **kwargs):
        super(WarpShear, self).__init__(**kwargs)

        assert len(ax) == 2 and len(ay) == 2
        assert ax[0] <= ax[1]
        assert ay[0] <= ay[1]
        self.ax = tuple(ax)
        self.ay = tuple(ay)

    @staticmethod
    def build_matrix(angles, img_size):
        w, h = img_size
        ax, ay = math.radians(angles[0]), math.radians(angles[1])

        C = np.eye(3)
        C[0, 2] = -w / 2
        C[1, 2] = -h / 2

        S = np.eye(3)
        S[0, 1] = math.tan(ax)
        S[1, 0] = math.tan(ay)

        CI = np.eye(3)
        CI[0, 2] = w / 2
        CI[1, 2] = h / 2

        return CI @ S @ C

    def __call__(self, data_dict):
        if 'warp_size' in data_dict.keys():
            size = data_dict['warp_size']
        else:
            size = get_image_size(data_dict['image'])

        shear = (random.uniform(*self.ax), random.uniform(*self.ay))

        M = self.build_matrix(shear, size)

        if self.expand:
            E, new_size = calc_expand_size_and_matrix(M, size)
            M = E @ M
            size = new_size

        data_dict['warp_tmp_matrix'] = M
        data_dict['warp_tmp_size'] = size
        data_dict = super(WarpShear, self).__call__(data_dict)

        return data_dict

    def __repr__(self):
        return 'WarpShear(ax={}, ay={}, {})'.format(self.ax, self.ay, super(WarpShear, self).__repr__())


@INTERNODE.register_module()
class WarpTranslate(WarpInternode):
    def __init__(self, rw, rh, **kwargs):
        super(WarpTranslate, self).__init__(**kwargs)

        assert len(rw) == 2 and len(rh) == 2
        assert rw[0] <= rw[1]
        assert rh[0] <= rh[1]
        self.rw = tuple(rw)
        self.rh = tuple(rh)

    @staticmethod
    def build_matrix(translations):
        T = np.eye(3)
        T[0, 2] = translations[0]
        T[1, 2] = translations[1]
        return T

    def __call__(self, data_dict):
        if 'warp_size' in data_dict.keys():
            size = data_dict['warp_size']
        else:
            size = get_image_size(data_dict['image'])

        min_dx = self.rw[0] * size[0]
        min_dy = self.rh[0] * size[1]
        max_dx = self.rw[1] * size[0]
        max_dy = self.rh[1] * size[1]
        translations = (np.round(random.uniform(min_dx, max_dx)),
                        np.round(random.uniform(min_dy, max_dy)))

        M = self.build_matrix(translations)

        data_dict['warp_tmp_matrix'] = M
        data_dict['warp_tmp_size'] = size
        data_dict = super(WarpTranslate, self).__call__(data_dict)

        return data_dict

    def __repr__(self):
        return 'WarpTranslate(rw={}, rh={}, {})'.format(self.rw, self.rh, super(WarpTranslate, self).__repr__())
