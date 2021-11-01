import cv2
import numpy as np
from ..base_internode import BaseInternode
from utils.point_tools import calc_pitch_yaw_roll


__all__ = ['AttributeSelector', 'CalcEulerAngles']


class AttributeSelector(BaseInternode):
    def __init__(self, indices):
        self.indices = list(indices)

    def __call__(self, data_dict):
        data_dict['attribute'] = data_dict['attribute'][self.indices]
        return data_dict

    def __repr__(self):
        return 'AttributeSelector(indices={})'.format(tuple(self.indices))

    def rper(self):
        return 'AttributeSelector(not available)'


class CalcEulerAngles(BaseInternode):
    def __init__(self, tp, path):
        self.tp = list(tp)
        self.path = path
        self.landmarks_3D = np.loadtxt(self.path)
        assert len(self.tp) == len(self.landmarks_3D)

    def __call__(self, data_dict):
        data_dict['euler_angle'] = calc_pitch_yaw_roll(data_dict['point'][self.tp], self.landmarks_3D).astype(np.float32)[0]
        return data_dict

    def __repr__(self):
        return 'CalcEulerAngles(tp={}, path={})'.format(tuple(self.tp), self.path)

    def rper(self):
        return 'CalcEulerAngles(not available)'
