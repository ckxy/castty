from ..base_internode import BaseInternode


__all__ = ['EraseContour']


class EraseContour(BaseInternode):
    def __call__(self, data_dict):
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
