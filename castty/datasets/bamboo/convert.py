import torch
import numpy as np
from PIL import Image
from .builder import INTERNODE
from .mixin import DataAugMixin
from ..utils.common import is_pil
from .base_internode import BaseInternode
from torchvision.transforms.functional import to_tensor, to_pil_image


__all__ = ['ToTensor', 'ToPILImage', 'ToCV2Image']


def cv22pil(image, meta=None, **kwargs):
    assert not is_pil(image)
    image = Image.fromarray(image)
    return image, meta


def pil2cv2(image, meta=None, **kwargs):
    assert is_pil(image)
    image = np.array(image)
    return image, meta


@INTERNODE.register_module()
class ToTensor(DataAugMixin, BaseInternode):
    def __init__(self, tag_mapping=dict(image=['image'], mask=['mask']), **kwargs):
        forward_mapping = dict(
            image=self.forward_image,
            mask=self.forward_mask
        )
        backward_mapping = dict(
            image=self.backward_image,
            mask=self.backward_mask
        )
        super(ToTensor, self).__init__(tag_mapping, forward_mapping, backward_mapping, **kwargs)

    def forward_image(self, image, meta, **kwargs):
        assert is_pil(image)
        image = to_tensor(image)
        return image, meta

    def forward_mask(self, mask, meta=None, **kwargs):
        mask = torch.from_numpy(mask)
        return mask, meta

    def backward_image(self, image, meta=None, **kwargs):
        image = to_pil_image(image)
        return image, meta

    def backward_mask(self, mask, meta=None, **kwargs):
        mask = mask.detach().cpu().numpy().astype(np.int32)
        return mask, meta

    def rper(self):
        return 'ToPILImage()'


@INTERNODE.register_module()
class ToPILImage(DataAugMixin, BaseInternode):
    def __init__(self, tag_mapping=dict(image=['image']), **kwargs):
        forward_mapping = dict(
            image=cv22pil
        )
        backward_mapping = dict(
            image=pil2cv2
        )
        super(ToPILImage, self).__init__(tag_mapping, forward_mapping, backward_mapping, **kwargs)

    def rper(self):
        return 'ToCV2Image()'


@INTERNODE.register_module()
class ToCV2Image(DataAugMixin, BaseInternode):
    def __init__(self, tag_mapping=dict(image=['image']), **kwargs):
        forward_mapping = dict(
            image=pil2cv2,
        )
        backward_mapping = dict(
            image=cv22pil,
        )
        super(ToCV2Image, self).__init__(tag_mapping, forward_mapping, backward_mapping, **kwargs)

    def rper(self):
        return 'ToPILImage()'


@INTERNODE.register_module()
class To1CHTensor(DataAugMixin, BaseInternode):
    def __init__(self, tag_mapping=dict(image=['image']), **kwargs):
        forward_mapping = dict(
            image=self.forward_image,
        )
        backward_mapping = dict(
            image=self.backward_image,
        )
        super(To1CHTensor, self).__init__(tag_mapping, forward_mapping, backward_mapping, **kwargs)

    def forward_image(self, image, meta, **kwargs):
        image = image[0].unsqueeze(0)
        return image, meta

    def backward_image(self, image, meta=None, **kwargs):
        image = image.repeat(3, 1, 1)
        return image, meta

    def rper(self):
        return 'To3CHTensor()'
