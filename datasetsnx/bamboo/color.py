import random
from PIL import Image
from .base_internode import BaseInternode
from torchvision.transforms.functional import normalize, to_grayscale, adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue


__all__ = ['BrightnessEnhancement', 'ContrastEnhancement', 'SaturationEnhancement', 'HueEnhancement', 'RandomGrayscale']


class BrightnessEnhancement(BaseInternode):
    def __init__(self, brightness, p=1):
        assert len(brightness) == 2
        assert brightness[1] >= brightness[0]
        assert brightness[0] > 0
        assert 0 < p <= 1

        self.brightness = brightness
        self.p = p
        # self.ColorJitter = torchvision.transforms.ColorJitter(self.brightness, 0, 0, 0)

    def __call__(self, data_dict):
        # data_dict = super(BrightnessEnhancement, self).__call__(data_dict)
        if random.random() < self.p:
            # data_dict['image'] = self.ColorJitter(data_dict['image'])
            factor = random.uniform(self.brightness[0], self.brightness[1])
            data_dict['image'] = adjust_brightness(data_dict['image'], factor)
        return data_dict

    def __repr__(self):
        return 'BrightnessEnhancement(p={}, brightness={})'.format(self.p, self.brightness)

    def rper(self):
        return 'BrightnessEnhancement(not available)'


class ContrastEnhancement(BaseInternode):
    def __init__(self, contrast, p=1):
        assert len(contrast) == 2
        assert contrast[1] >= contrast[0]
        assert contrast[0] > 0
        assert 0 < p <= 1

        self.contrast = contrast
        self.p = p
        # self.ColorJitter = torchvision.transforms.ColorJitter(0, self.contrast, 0, 0)

    def __call__(self, data_dict):
        # data_dict = super(ContrastEnhancement, self).__call__(data_dict)
        if random.random() < self.p:
            # data_dict['image'] = self.ColorJitter(data_dict['image'])
            factor = random.uniform(self.contrast[0], self.contrast[1])
            data_dict['image'] = adjust_contrast(data_dict['image'], factor)
        return data_dict

    def __repr__(self):
        return 'ContrastEnhancement(p={}, contrast={})'.format(self.p, self.contrast)

    def rper(self):
        return 'ContrastEnhancement(not available)'



class SaturationEnhancement(BaseInternode):
    def __init__(self, saturation, p=1):
        assert len(saturation) == 2
        assert saturation[1] >= saturation[0]
        assert saturation[0] > 0
        assert 0 < p <= 1

        self.saturation = saturation
        self.p = p
        # self.ColorJitter = torchvision.transforms.ColorJitter(0, 0, self.saturation, 0)

    def __call__(self, data_dict):
        # data_dict = super(SaturationEnhancement, self).__call__(data_dict)
        if random.random() < self.p:
            # data_dict['image'] = self.ColorJitter(data_dict['image'])
            factor = random.uniform(self.saturation[0], self.saturation[1])
            data_dict['image'] = adjust_saturation(data_dict['image'], factor)
        return data_dict

    def __repr__(self):
        return 'SaturationEnhancement(p={}, saturation={})'.format(self.p, self.saturation)

    def rper(self):
        return 'SaturationEnhancement(not available)'


class HueEnhancement(BaseInternode):
    def __init__(self, hue, p=1):
        assert len(hue) == 2
        assert hue[1] >= hue[0]
        assert hue[0] >= -0.5 and hue[1] <= 0.5
        assert 0 < p <= 1

        self.hue = hue
        self.p = p
        # self.ColorJitter = torchvision.transforms.ColorJitter(0, 0, 0, self.hue)

    def __call__(self, data_dict):
        # data_dict = super(HueEnhancement, self).__call__(data_dict)
        if random.random() < self.p:
            # data_dict['image'] = self.ColorJitter(data_dict['image'])
            factor = random.uniform(self.hue[0], self.hue[1])
            data_dict['image'] = adjust_hue(data_dict['image'], factor)
        return data_dict

    def __repr__(self):
        return 'HueEnhancement(p={}, hue={})'.format(self.p, self.hue)

    def rper(self):
        return 'HueEnhancement(not available)'


class RandomGrayscale(BaseInternode):
    def __init__(self, ch, p=1):
        assert 0 < p <= 1
        assert self.ch == 1 or self.ch == 3
        self.p = p
        self.ch = ch

    def __call__(self, data_dict):
        # data_dict = super(RandomGrayscale, self).__call__(data_dict)
        if random.random() < self.p:
            data_dict['image'] = to_grayscale(data_dict['image'], num_output_channels=self.ch)
        return data_dict

    def __repr__(self):
        return 'RandomGrayscale(p={}, ch={})'.format(self.p, self.ch)

    def rper(self):
        return 'RandomGrayscale(not available)'
