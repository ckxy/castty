from ..base_internode import BaseInternode


__all__ = ['NormCoor']


class NormCoor(BaseInternode):
    def __init__(self, size=None, no_forward=False):
        self.size = size
        self.no_forward = no_forward

    def __call__(self, data_dict):
        if self.no_forward:
            return data_dict

        if self.size is None:
            w, h = data_dict['image'].size
        else:
            w, h = self.size

        if 'bbox' in data_dict.keys():
            data_dict['bbox'][:, 0] /= w
            data_dict['bbox'][:, 1] /= h
            data_dict['bbox'][:, 2] /= w
            data_dict['bbox'][:, 3] /= h

        if 'point' in data_dict.keys():
            data_dict['point'][:, 0] /= w
            data_dict['point'][:, 1] /= h

        return data_dict

    def reverse(self, **kwargs):
        if self.size is None or self.no_forward:
            h, w = kwargs['ori_size']
            h, w = int(h), int(w)
        else:
            w, h = self.size

        if 'bbox' in kwargs.keys():
            kwargs['bbox'][:, 0] *= w
            kwargs['bbox'][:, 1] *= h
            kwargs['bbox'][:, 2] *= w
            kwargs['bbox'][:, 3] *= h

        if 'point' in kwargs.keys():
            kwargs['point'][:, 0] *= w
            kwargs['point'][:, 1] *= h

        return kwargs

    def __repr__(self):
        if self.no_forward:
            return 'CoorNorm(not available)'
        else:
            if self.size is None:
                return 'CoorNorm(adaptive)'
            else:
                return 'CoorNorm(size={})'.format(self.size)

    def rper(self):
        if self.no_forward:
            return 'CoorNorm(backward adaptive)'
        else:
            if self.size is None:
                return 'CoorNorm(adaptive)'
            else:
                return 'CoorNorm(size={})'.format(self.size)
