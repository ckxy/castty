import numpy as np
from PIL import Image
from .base_internode import BaseInternode
from torchvision.transforms.functional import pad
from .utils.warp_tools import clip_bbox, filter_bbox


__all__ = ['Resize', 'ResizeAndPadding']


class Resize(BaseInternode):
    def __init__(self, size, keep_ratio=False, short=True):
        assert len(size) == 2
        assert size[0] > 0 and size[1] > 0

        self.size = size
        self.keep_ratio = keep_ratio
        if self.keep_ratio:
            self.short = short
        else:
            self.short = None

    def calc_scale(self, size):
        w, h = size
        tw, th = self.size
        rw, rh = tw / w, th / h

        if self.keep_ratio:
            if self.short:
                r = min(rh, rw)
                scale = (r, r)
            else:
                r = max(rh, rw)
                scale = (r, r)
        else:
            scale = (rw, rh)
        return scale

    def __call__(self, data_dict):
        scale = self.calc_scale(data_dict['image'].size)

        w, h = data_dict['image'].size
        nw = int(scale[0] * w)
        nh = int(scale[1] * h)

        data_dict['image'] = data_dict['image'].resize((nw, nh), Image.BILINEAR)

        if 'bbox' in data_dict.keys():
            data_dict['bbox'][:, 0] *= scale[0]
            data_dict['bbox'][:, 1] *= scale[1]
            data_dict['bbox'][:, 2] *= scale[0]
            data_dict['bbox'][:, 3] *= scale[1]

        if 'point' in data_dict.keys():
            data_dict['point'][:, 0] *= scale[0]
            data_dict['point'][:, 1] *= scale[1]

        if 'mask' in data_dict.keys():
            data_dict['mask'] = data_dict['mask'].resize((nw, nh), Image.NEAREST)

        # if 'polygon' in data_dict.keys():
        #     g, p, _ = data_dict['polygon'].shape
        #     data_dict['polygon'][..., :2] = self._resize_point(data_dict['polygon'][..., :2].reshape(-1, 2)).reshape(g, p, 2)

        return data_dict

    def __repr__(self):
        return 'Resize(size={}, keep_ratio={}, short={})'.format(self.size, self.keep_ratio, self.short)

    def rper(self):
        return 'Resize(not available)'


class ResizeAndPadding(BaseInternode):
    def __init__(self, size, keep_ratio=True, warp=False):
        assert len(size) == 2
        assert size[0] > 0 and size[1] > 0

        self.size = size
        self.keep_ratio = keep_ratio
        self.warp = warp

    def build_matrix(self, img_size, scale):
        w, h = img_size

        C = np.eye(3)
        C[0, 2] = w / 2
        C[1, 2] = h / 2

        R = np.eye(3)
        R[0, 0] = 1 / scale[0]
        R[1, 1] = 1 / scale[1]

        CI = np.eye(3)
        # l = int(min(w, h) * scale[0])
        # print(l)
        # exit()
        CI[0, 2] = -self.size[0] / 2 + (self.size[0] - int(scale[0] * w)) / 2
        CI[1, 2] = -self.size[1] / 2 + (self.size[1] - int(scale[1] * h)) / 2

        M = C @ R @ CI

        return M[:2].flatten().tolist()

    def build_inverse_matrix(self, img_size, scale):
        w, h = img_size

        C = np.eye(3)
        C[0, 2] = -w / 2
        C[1, 2] = -h / 2

        R = np.eye(3)
        R[0, 0] = scale[0]
        R[1, 1] = scale[1]

        CI = np.eye(3)
        CI[0, 2] = self.size[0] / 2 - (self.size[0] - int(scale[0] * w)) / 2
        CI[1, 2] = self.size[1] / 2 - (self.size[1] - int(scale[1] * h)) / 2

        M = CI @ R @ C

        return M[:2].flatten().tolist()

    def calc_scale(self, size):
        w, h = size
        tw, th = self.size
        rw, rh = tw / w, th / h

        if self.keep_ratio:
            r = min(rh, rw)
            return (r, r)
        else:
            return (rw, rh)

    def __call__(self, data_dict):
        scale = self.calc_scale(data_dict['image'].size)

        w, h = data_dict['image'].size
        nw = int(scale[0] * w)
        nh = int(scale[1] * h)

        l = max(nw, nh)

        right = self.size[0] - nw
        bottom = self.size[1] - nh

        # print(right, bottom)

        if self.warp:
            matrix = self.build_matrix((w, h), scale)
            data_dict['image'] = data_dict['image'].transform(self.size, Image.AFFINE, matrix, Image.BILINEAR)
        else:
            data_dict['image'] = data_dict['image'].resize((nw, nh), Image.BILINEAR)
            data_dict['image'] = pad(data_dict['image'], (0, 0, right, bottom), 0, 'constant')

        if 'mask' in data_dict.keys():
            if self.warp:
                data_dict['mask'] = data_dict['mask'].transform(self.size, Image.AFFINE, matrix, Image.NEAREST)
            else:
                data_dict['mask'] = data_dict['mask'].resize((nw, nh), Image.NEAREST)
                data_dict['mask'] = pad(data_dict['mask'], (0, 0, right, bottom), 0, 'constant')

        if 'bbox' in data_dict.keys():
            data_dict['bbox'][:, 0] = data_dict['bbox'][:, 0] * scale[0]
            data_dict['bbox'][:, 1] = data_dict['bbox'][:, 1] * scale[1]
            data_dict['bbox'][:, 2] = data_dict['bbox'][:, 2] * scale[0]
            data_dict['bbox'][:, 3] = data_dict['bbox'][:, 3] * scale[1]

        if 'point' in data_dict.keys():
            data_dict['point'][:, 0] = data_dict['point'][:, 0] * scale[0]
            data_dict['point'][:, 1] = data_dict['point'][:, 1] * scale[1]

        if 'quad' in data_dict.keys():
            # print(scale)
            data_dict['quad'][..., 0] = data_dict['quad'][..., 0] * scale[0]
            data_dict['quad'][..., 1] = data_dict['quad'][..., 1] * scale[1]

        return data_dict

    def reverse(self, **kwargs):
        if 'training' in kwargs.keys() and kwargs['training']:
            return kwargs

        h, w = kwargs['ori_size']
        h, w = int(h), int(w)
        scale = self.calc_scale((w, h))

        # print(kwargs['ori_size'], kwargs['image'].size, scale)

        nw = int(scale[0] * w)
        nh = int(scale[1] * h)
        l = max(nw, nh)

        # print(nw, nh)

        right = nw
        bottom = nh
        # print(left, top, right, bottom)
        # exit()

        if 'image' in kwargs.keys():
            if self.warp:
                matrix = self.build_inverse_matrix((w, h), scale)
                kwargs['image'] = kwargs['image'].transform((w, h), Image.AFFINE, matrix, Image.BILINEAR)
            else:
                kwargs['image'] = kwargs['image'].crop((0, 0, right, bottom))
                # print(kwargs['image'])
                kwargs['image'] = kwargs['image'].resize((w, h), Image.BILINEAR)
                # print(kwargs['image'])
        # exit()

        if 'mask' in kwargs.keys():
            if self.warp:
                kwargs['mask'] = kwargs['mask'].transform((w, h), Image.AFFINE, matrix, Image.NEAREST)
            else:
                kwargs['mask'] = kwargs['mask'].crop((0, 0, right, bottom))
                kwargs['mask'] = kwargs['mask'].resize((w, h), Image.NEAREST)

        if 'bbox' in kwargs.keys():
            kwargs['bbox'][:, 0] = kwargs['bbox'][:, 0] / scale[0]
            kwargs['bbox'][:, 1] = kwargs['bbox'][:, 1] / scale[1]
            kwargs['bbox'][:, 2] = kwargs['bbox'][:, 2] / scale[0]
            kwargs['bbox'][:, 3] = kwargs['bbox'][:, 3] / scale[1]

            boxes = kwargs['bbox'][:, :4]
            other = kwargs['bbox'][:, 4:]
            boxes = clip_bbox(boxes, (w, h))
            keep = filter_bbox(boxes)

            kwargs['bbox'] = np.concatenate((boxes, other), axis=-1)[keep]

        if 'point' in kwargs.keys():
            kwargs['point'][:, 0] = kwargs['point'][:, 0] / scale[0]
            kwargs['point'][:, 1] = kwargs['point'][:, 1] / scale[1]

        if 'quad' in kwargs.keys():
            kwargs['quad'][..., 0] = kwargs['quad'][..., 0] / scale[0]
            kwargs['quad'][..., 1] = kwargs['quad'][..., 1] / scale[1]

        return kwargs

    def __repr__(self):
        return 'ResizeAndPadding(size={}, keep_ratio={}, warp={})'.format(self.size, self.keep_ratio, self.warp)

    def rper(self):
        return self.__repr__()