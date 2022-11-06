import copy
from .builder import INTERNODE
from collections import Iterable
from .base_internode import BaseInternode

from PIL import Image,ImageFont,ImageDraw, ImageFilter
from math import pi, cos, sin, tan
from random import randint


__all__ = ['SealGen']


@INTERNODE.register_module()
class SealGen(BaseInternode):
    def __init__(self, tags):
        pass

    def forward(self, data_dict):
        for tag in self.tags:
            data_dict.pop(tag)
        return data_dict

    def __repr__(self):
        return 'SealGen()'.format()
