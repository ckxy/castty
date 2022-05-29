import cv2
import random
import numpy as np
from PIL import Image
from .builder import INTERNODE
from .base_internode import BaseInternode
from ..utils.warp_tools import is_pil
from torchvision.transforms.functional import normalize, to_grayscale, adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue


__all__ = ['BrightnessEnhancement', 'ContrastEnhancement', 'SaturationEnhancement', 'HueEnhancement', 'ToGrayscale']


@INTERNODE.register_module()
class BrightnessEnhancement(BaseInternode):
    def __init__(self, brightness, **kwargs):
        assert len(brightness) == 2
        assert brightness[1] >= brightness[0]
        assert brightness[0] > 0

        self.brightness = brightness

    def __call__(self, data_dict):
        factor = random.uniform(self.brightness[0], self.brightness[1])

        if is_pil(data_dict['image']):
            data_dict['image'] = adjust_brightness(data_dict['image'], factor)
        else:
            data_dict['image'] = Image.fromarray(data_dict['image'])
            data_dict['image'] = adjust_brightness(data_dict['image'], factor)
            data_dict['image'] = np.asarray(data_dict['image'])

        return data_dict

    def __repr__(self):
        return 'BrightnessEnhancement(brightness={})'.format(self.brightness)


@INTERNODE.register_module()
class ContrastEnhancement(BaseInternode):
    def __init__(self, contrast, **kwargs):
        assert len(contrast) == 2
        assert contrast[1] >= contrast[0]
        assert contrast[0] > 0

        self.contrast = contrast

    def __call__(self, data_dict):
        factor = random.uniform(self.contrast[0], self.contrast[1])

        if is_pil(data_dict['image']):
            data_dict['image'] = adjust_contrast(data_dict['image'], factor)
        else:
            data_dict['image'] = Image.fromarray(data_dict['image'])
            data_dict['image'] = adjust_contrast(data_dict['image'], factor)
            data_dict['image'] = np.asarray(data_dict['image'])

        return data_dict

    def __repr__(self):
        return 'ContrastEnhancement(contrast={})'.format(self.contrast)


@INTERNODE.register_module()
class SaturationEnhancement(BaseInternode):
    def __init__(self, saturation, **kwargs):
        assert len(saturation) == 2
        assert saturation[1] >= saturation[0]
        assert saturation[0] > 0

        self.saturation = saturation

    def __call__(self, data_dict):
        factor = random.uniform(self.saturation[0], self.saturation[1])

        if is_pil(data_dict['image']):
            data_dict['image'] = adjust_saturation(data_dict['image'], factor)
        else:
            data_dict['image'] = Image.fromarray(data_dict['image'])
            data_dict['image'] = adjust_saturation(data_dict['image'], factor)
            data_dict['image'] = np.asarray(data_dict['image'])

        return data_dict

    def __repr__(self):
        return 'SaturationEnhancement(saturation={})'.format(self.saturation)


@INTERNODE.register_module()
class HueEnhancement(BaseInternode):
    def __init__(self, hue, **kwargs):
        assert len(hue) == 2
        assert hue[1] >= hue[0]
        assert hue[0] >= -0.5 and hue[1] <= 0.5

        self.hue = hue

    def __call__(self, data_dict):
        factor = random.uniform(self.hue[0], self.hue[1])

        if is_pil(data_dict['image']):
            data_dict['image'] = adjust_hue(data_dict['image'], factor)
        else:
            data_dict['image'] = Image.fromarray(data_dict['image'])
            data_dict['image'] = adjust_hue(data_dict['image'], factor)
            data_dict['image'] = np.asarray(data_dict['image'])

        return data_dict

    def __repr__(self):
        return 'HueEnhancement(hue={})'.format(self.hue)


@INTERNODE.register_module()
class ToGrayscale(BaseInternode):
    def __call__(self, data_dict):
        if is_pil(data_dict['image']):
            data_dict['image'] = to_grayscale(data_dict['image'], num_output_channels=3)
        else:
            # print(data_dict['image'].shape, 'a')
            data_dict['image'] = cv2.cvtColor(data_dict['image'], cv2.COLOR_BGR2GRAY)
            data_dict['image'] = data_dict['image'][..., np.newaxis]
            data_dict['image'] = np.repeat(data_dict['image'], 3, axis=-1)
            # print(data_dict['image'].shape, 'b')
        return data_dict
