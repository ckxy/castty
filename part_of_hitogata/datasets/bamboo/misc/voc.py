from ..base_internode import BaseInternode
from ..builder import INTERNODE


__all__ = ['EraseContour']


@INTERNODE.register_module()
class EraseContour(BaseInternode):
    def __call__(self, data_dict):
        import torch
        print(torch.unique(data_dict['mask']))
        exit()
        data_dict['mask'][data_dict['mask'] == 255] = -1  # Ignore contour
        return data_dict

    def reverse(self, **kwargs):
        if 'mask' in kwargs.keys():
            kwargs['mask'][kwargs['mask'] == -1] = 255
        return kwargs

    def __repr__(self):
        return 'EraseContour()'

    def rper(self):
        return 'DrawContour()'
