import cv2
import random
import numpy as np
from PIL import Image
from .builder import INTERNODE
from ..utils.common import is_pil
from .base_internode import BaseInternode
from torchvision.transforms.functional import normalize, to_grayscale, adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue


__all__ = ['BrightnessEnhancement', 'ContrastEnhancement', 'SaturationEnhancement', 'HueEnhancement', 'ToGrayscale']


@INTERNODE.register_module()
class BrightnessEnhancement(BaseInternode):
    def __init__(self, brightness, **kwargs):
        assert len(brightness) == 2
        assert brightness[1] >= brightness[0]
        assert brightness[0] > 0

        self.brightness = brightness

        super(BrightnessEnhancement, self).__init__(**kwargs)

    def calc_intl_param_forward(self, data_dict):
        data_dict['intl_factor'] = random.uniform(self.brightness[0], self.brightness[1])
        return data_dict

    def forward_image(self, data_dict):
        target_tag = data_dict['intl_base_target_tag']
        
        if is_pil(data_dict[target_tag]):
            data_dict[target_tag] = adjust_brightness(data_dict[target_tag], data_dict['intl_factor'])
        else:
            data_dict[target_tag] = Image.fromarray(data_dict[target_tag])
            data_dict[target_tag] = adjust_brightness(data_dict[target_tag], data_dict['intl_factor'])
            data_dict[target_tag] = np.array(data_dict[target_tag])

        return data_dict

    def erase_intl_param_forward(self, data_dict):
        data_dict.pop('intl_factor')
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

        super(ContrastEnhancement, self).__init__(**kwargs)

    def calc_intl_param_forward(self, data_dict):
        data_dict['intl_factor'] = random.uniform(self.contrast[0], self.contrast[1])
        return data_dict

    def forward_image(self, data_dict):
        target_tag = data_dict['intl_base_target_tag']

        if is_pil(data_dict[target_tag]):
            data_dict[target_tag] = adjust_contrast(data_dict[target_tag], data_dict['intl_factor'])
        else:
            data_dict[target_tag] = Image.fromarray(data_dict[target_tag])
            data_dict[target_tag] = adjust_contrast(data_dict[target_tag], data_dict['intl_factor'])
            data_dict[target_tag] = np.array(data_dict[target_tag])

        return data_dict

    def erase_intl_param_forward(self, data_dict):
        data_dict.pop('intl_factor')
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

        super(SaturationEnhancement, self).__init__(**kwargs)

    def calc_intl_param_forward(self, data_dict):
        data_dict['intl_factor'] = random.uniform(self.saturation[0], self.saturation[1])
        return data_dict

    def forward_image(self, data_dict):
        target_tag = data_dict['intl_base_target_tag']

        if is_pil(data_dict[target_tag]):
            data_dict[target_tag] = adjust_saturation(data_dict[target_tag], data_dict['intl_factor'])
        else:
            data_dict[target_tag] = Image.fromarray(data_dict[target_tag])
            data_dict[target_tag] = adjust_saturation(data_dict[target_tag], data_dict['intl_factor'])
            data_dict[target_tag] = np.array(data_dict[target_tag])

        return data_dict

    def erase_intl_param_forward(self, data_dict):
        data_dict.pop('intl_factor')
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

        super(HueEnhancement, self).__init__(**kwargs)

    def calc_intl_param_forward(self, data_dict):
        data_dict['intl_factor'] = random.uniform(self.hue[0], self.hue[1])
        return data_dict

    def forward_image(self, data_dict):
        target_tag = data_dict['intl_base_target_tag']

        if is_pil(data_dict[target_tag]):
            data_dict[target_tag] = adjust_hue(data_dict[target_tag], data_dict['intl_factor'])
        else:
            data_dict[target_tag] = Image.fromarray(data_dict[target_tag])
            data_dict[target_tag] = adjust_hue(data_dict[target_tag], data_dict['intl_factor'])
            data_dict[target_tag] = np.array(data_dict[target_tag])

        return data_dict

    def erase_intl_param_forward(self, data_dict):
        data_dict.pop('intl_factor')
        return data_dict

    def __repr__(self):
        return 'HueEnhancement(hue={})'.format(self.hue)


@INTERNODE.register_module()
class ToGrayscale(BaseInternode):
    def forward_image(self, data_dict):
        target_tag = data_dict['intl_base_target_tag']

        if is_pil(data_dict[target_tag]):
            data_dict[target_tag] = to_grayscale(data_dict[target_tag], num_output_channels=3)
        else:
            data_dict[target_tag] = cv2.cvtColor(data_dict[target_tag], cv2.COLOR_BGR2GRAY)
            data_dict[target_tag] = data_dict[target_tag][..., np.newaxis]
            data_dict[target_tag] = np.repeat(data_dict[target_tag], 3, axis=-1)
        return data_dict
