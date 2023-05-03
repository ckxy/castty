import cv2
import random
import numpy as np
from PIL import Image
from .builder import INTERNODE
from .mixin import DataAugMixin
from ..utils.common import is_pil
from .base_internode import BaseInternode
from torchvision.transforms.functional import normalize, to_grayscale, adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue


__all__ = ['BrightnessEnhancement', 'ContrastEnhancement', 'SaturationEnhancement', 'HueEnhancement', 'ToGrayscale']


@INTERNODE.register_module()
class BrightnessEnhancement(DataAugMixin, BaseInternode):
    def __init__(self, brightness, tag_mapping=dict(image=['image']), **kwargs):
        assert len(brightness) == 2
        assert brightness[1] >= brightness[0]
        assert brightness[0] > 0

        self.brightness = brightness

        forward_mapping = dict(
            image=self.forward_image
        )
        backward_mapping = dict()
        super(BrightnessEnhancement, self).__init__(tag_mapping, forward_mapping, backward_mapping, **kwargs)

    def calc_intl_param_forward(self, data_dict):
        intl_brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
        return dict(intl_brightness_factor=intl_brightness_factor)

    def forward_image(self, image, meta, intl_brightness_factor, **kwargs):
        if is_pil(image):
            image = adjust_brightness(image, intl_brightness_factor)
        else:
            image = Image.fromarray(image)
            image = adjust_brightness(image, intl_brightness_factor)
            image = np.array(image)

        return image, meta

    def __repr__(self):
        return 'BrightnessEnhancement(brightness={})'.format(self.brightness)


@INTERNODE.register_module()
class ContrastEnhancement(DataAugMixin, BaseInternode):
    def __init__(self, contrast, tag_mapping=dict(image=['image']), **kwargs):
        assert len(contrast) == 2
        assert contrast[1] >= contrast[0]
        assert contrast[0] > 0

        self.contrast = contrast

        forward_mapping = dict(
            image=self.forward_image
        )
        backward_mapping = dict()
        super(ContrastEnhancement, self).__init__(tag_mapping, forward_mapping, backward_mapping, **kwargs)

    def calc_intl_param_forward(self, data_dict):
        intl_contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
        return dict(intl_contrast_factor=intl_contrast_factor)

    def forward_image(self, image, meta, intl_contrast_factor, **kwargs):
        if is_pil(image):
            image = adjust_contrast(image, intl_contrast_factor)
        else:
            image = Image.fromarray(image)
            image = adjust_contrast(image, intl_contrast_factor)
            image = np.array(image)

        return image, meta

    def __repr__(self):
        return 'ContrastEnhancement(contrast={})'.format(self.contrast)


@INTERNODE.register_module()
class SaturationEnhancement(DataAugMixin, BaseInternode):
    def __init__(self, saturation, tag_mapping=dict(image=['image']), **kwargs):
        assert len(saturation) == 2
        assert saturation[1] >= saturation[0]
        assert saturation[0] > 0

        self.saturation = saturation

        forward_mapping = dict(
            image=self.forward_image
        )
        backward_mapping = dict()
        super(SaturationEnhancement, self).__init__(tag_mapping, forward_mapping, backward_mapping, **kwargs)

    def calc_intl_param_forward(self, data_dict):
        intl_saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
        return dict(intl_saturation_factor=intl_saturation_factor)

    def forward_image(self, image, meta, intl_saturation_factor, **kwargs):
        if is_pil(image):
            image = adjust_saturation(image, intl_saturation_factor)
        else:
            image = Image.fromarray(image)
            image = adjust_saturation(image, intl_saturation_factor)
            image = np.array(image)

        return image, meta

    def __repr__(self):
        return 'SaturationEnhancement(saturation={})'.format(self.saturation)


@INTERNODE.register_module()
class HueEnhancement(DataAugMixin, BaseInternode):
    def __init__(self, hue, tag_mapping=dict(image=['image']), **kwargs):
        assert len(hue) == 2
        assert hue[1] >= hue[0]
        assert hue[0] >= -0.5 and hue[1] <= 0.5

        self.hue = hue

        forward_mapping = dict(
            image=self.forward_image
        )
        backward_mapping = dict()
        super(HueEnhancement, self).__init__(tag_mapping, forward_mapping, backward_mapping, **kwargs)

    def calc_intl_param_forward(self, data_dict):
        intl_hue_factor = random.uniform(self.hue[0], self.hue[1])
        return dict(intl_hue_factor=intl_hue_factor)

    def forward_image(self, image, meta, intl_hue_factor, **kwargs):
        if is_pil(image):
            image = adjust_hue(image, intl_hue_factor)
        else:
            image = Image.fromarray(image)
            image = adjust_hue(image, intl_hue_factor)
            image = np.array(image)

        return image, meta

    def __repr__(self):
        return 'HueEnhancement(hue={})'.format(self.hue)


@INTERNODE.register_module()
class ToGrayscale(DataAugMixin, BaseInternode):
    def __init__(self, tag_mapping=dict(image=['image']), **kwargs):
        forward_mapping = dict(
            image=self.forward_image
        )
        backward_mapping = dict()
        super(ToGrayscale, self).__init__(tag_mapping, forward_mapping, backward_mapping, **kwargs)

    def forward_image(self, image, meta, **kwargs):
        if is_pil(image):
            image = to_grayscale(image, num_output_channels=3)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image[..., np.newaxis]
            image = np.repeat(image, 3, axis=-1)
        return image, meta
