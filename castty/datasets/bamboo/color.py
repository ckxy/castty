import cv2
import random
import numpy as np
from .builder import INTERNODE
from .mixin import DataAugMixin
from PIL import Image, ImageEnhance
from ..utils.common import is_pil, is_cv2
from .base_internode import BaseInternode
from torchvision.transforms.functional import normalize, rgb_to_grayscale, adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue


__all__ = ['BrightnessEnhancement', 'ContrastEnhancement', 'SaturationEnhancement', 'HueEnhancement', 'ToGrayscale']


def enhance_bcs(image, factor, mode='Brightness'):
    assert mode in ['Brightness', 'Contrast', 'Saturation']

    if mode == 'Brightness':
        enhancer = ImageEnhance.Brightness(image)
    elif mode == 'Contrast':
        enhancer = ImageEnhance.Contrast(image)
    else:
        enhancer = ImageEnhance.Color(image)

    return enhancer.enhance(factor)


# copy from torchvision/transforms/_functional_pil.py
def enhance_h(img: Image.Image, hue_factor: float) -> Image.Image:
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError(f"hue_factor ({hue_factor}) is not in [-0.5, 0.5].")

    if not is_pil(img):
        raise TypeError(f"img should be PIL Image. Got {type(img)}")

    input_mode = img.mode
    if input_mode in {"L", "1", "I", "F"}:
        return img

    h, s, v = img.convert("HSV").split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over="ignore"):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, "L")

    img = Image.merge("HSV", (h, s, v)).convert(input_mode)
    return img


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
        DataAugMixin.__init__(self, tag_mapping, forward_mapping, backward_mapping)
        BaseInternode.__init__(self, **kwargs)

    def calc_intl_param_forward(self, data_dict):
        intl_brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
        return dict(intl_brightness_factor=intl_brightness_factor)

    def forward_image(self, image, meta, intl_brightness_factor, **kwargs):
        if is_pil(image):
            # image = adjust_brightness(image, intl_brightness_factor)
            image = enhance_bcs(image, intl_brightness_factor, 'Brightness')
        elif is_cv2(image):
            image = Image.fromarray(image)
            # image = adjust_brightness(image, intl_brightness_factor)
            image = enhance_bcs(image, intl_brightness_factor, 'Brightness')
            image = np.array(image)
        else:
            image = adjust_brightness(image, intl_brightness_factor)

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
        DataAugMixin.__init__(self, tag_mapping, forward_mapping, backward_mapping)
        BaseInternode.__init__(self, **kwargs)

    def calc_intl_param_forward(self, data_dict):
        intl_contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
        return dict(intl_contrast_factor=intl_contrast_factor)

    def forward_image(self, image, meta, intl_contrast_factor, **kwargs):
        if is_pil(image):
            # image = adjust_contrast(image, intl_contrast_factor)
            image = enhance_bcs(image, intl_contrast_factor, 'Contrast')
        elif is_cv2(image):
            image = Image.fromarray(image)
            # image = adjust_contrast(image, intl_contrast_factor)
            image = enhance_bcs(image, intl_contrast_factor, 'Contrast')
            image = np.array(image)
        else:
            image = adjust_contrast(image, intl_contrast_factor)

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
        DataAugMixin.__init__(self, tag_mapping, forward_mapping, backward_mapping)
        BaseInternode.__init__(self, **kwargs)

    def calc_intl_param_forward(self, data_dict):
        intl_saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
        return dict(intl_saturation_factor=intl_saturation_factor)

    def forward_image(self, image, meta, intl_saturation_factor, **kwargs):
        if is_pil(image):
            # image = adjust_saturation(image, intl_saturation_factor)
            image = enhance_bcs(image, intl_saturation_factor, 'Saturation')
        elif is_cv2(image):
            image = Image.fromarray(image)
            # image = adjust_saturation(image, intl_saturation_factor)
            image = enhance_bcs(image, intl_saturation_factor, 'Saturation')
            image = np.array(image)
        else:
            image = adjust_saturation(image, intl_saturation_factor)

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
        DataAugMixin.__init__(self, tag_mapping, forward_mapping, backward_mapping)
        BaseInternode.__init__(self, **kwargs)

    def calc_intl_param_forward(self, data_dict):
        intl_hue_factor = random.uniform(self.hue[0], self.hue[1])
        return dict(intl_hue_factor=intl_hue_factor)

    def forward_image(self, image, meta, intl_hue_factor, **kwargs):
        if is_pil(image):
            # image = adjust_hue(image, intl_hue_factor)
            image = enhance_h(image, intl_hue_factor)
        elif is_cv2(image):
            image = Image.fromarray(image)
            # image = adjust_hue(image, intl_hue_factor)
            image = enhance_h(image, intl_hue_factor)
            image = np.array(image)
        else:
            image = adjust_hue(image, intl_hue_factor)

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
        DataAugMixin.__init__(self, tag_mapping, forward_mapping, backward_mapping)
        BaseInternode.__init__(self, **kwargs)

    def forward_image(self, image, meta, **kwargs):
        if is_pil(image):
            # image = rgb_to_grayscale(image, num_output_channels=3)
            image = image.convert("L")
            np_img = np.array(image, dtype=np.uint8)
            np_img = np.dstack([np_img, np_img, np_img])
            image = Image.fromarray(np_img, "RGB")
        elif is_cv2(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image[..., np.newaxis]
            image = np.repeat(image, 3, axis=-1)
        else:
            image = rgb_to_grayscale(image, num_output_channels=3)
        return image, meta
