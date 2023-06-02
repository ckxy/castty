import math
from ..builder import INTERNODE
from ..resize import ResizeInternode
from ..base_internode import BaseInternode
from ...utils.common import get_image_size
from ..pad import PaddingInternode, TAG_MAPPING
try:
    from mmengine.structures import BaseDataElement, LabelData
    from mmocr.structures import TextRecogDataSample
except:
    pass


__all__ = ['PadToWidth', 'PackTextRecogInputs']


@INTERNODE.register_module()
class RescaleToHeight(ResizeInternode):
    def __init__(self, height, min_width=None, max_width=None, width_divisor=1, tag_mapping=TAG_MAPPING, **kwargs):
        assert isinstance(height, int)
        assert isinstance(width_divisor, int)
        if min_width is not None:
            assert isinstance(min_width, int)
        if max_width is not None:
            assert isinstance(max_width, int)
        self.width_divisor = width_divisor
        self.height = height
        self.min_width = min_width
        self.max_width = max_width

        ResizeInternode.__init__(self, tag_mapping, **kwargs)

    def calc_scale_and_new_size(self, w, h):
        new_width = math.ceil(float(self.height) / h * w)
        if self.min_width is not None:
            new_width = max(self.min_width, new_width)
        if self.max_width is not None:
            new_width = min(self.max_width, new_width)

        if new_width % self.width_divisor != 0:
            new_width = round(
                new_width / self.width_divisor) * self.width_divisor
        # TODO replace up code after testing precision.
        # new_width = math.ceil(
        #     new_width / self.width_divisor) * self.width_divisor
        new_size = (new_width, self.height)
        scale = (new_width / w, self.height / h)

        return scale, new_size


@INTERNODE.register_module()
class PadToWidth(PaddingInternode):
    def __init__(self, width, fill=(0, 0, 0), padding_mode='constant', tag_mapping=TAG_MAPPING, **kwargs):
        assert width > 0
        self.width = width

        PaddingInternode.__init__(self, fill=fill, padding_mode=padding_mode, tag_mapping=tag_mapping, **kwargs)

    def calc_padding(self, w, h):
        return 0, 0, max(0, self.width - w), 0

    def forward_rest(self, data_dict, **kwargs):
        _, _, right, _ = kwargs['intl_padding']
        ori_width, _ = get_image_size(data_dict['image'])
        ori_width = ori_width - right
        data_dict['ocrreg_valid_ratio'] = min(1.0, 1.0 * ori_width / self.width)
        return data_dict

    def __repr__(self):
        return 'PadToWidthForMMOCR(width={}, fill={}, padding_mode={})'.format(self.width, self.fill, self.padding_mode)


@INTERNODE.register_module()
class PackTextRecogInputs(BaseInternode):
    def forward(self, data_dict, **kwargs):
        data_sample = TextRecogDataSample()
        gt_text = LabelData()

        gt_text.item = data_dict['seq']
        data_sample.gt_text = gt_text

        ow, oh = data_dict['image_meta']['ori_size']
        w, h = get_image_size(data_dict['image'])

        img_meta = dict(
            img_path=data_dict['image_meta']['path'],
            ori_shape=(oh, ow),
            img_shape=(h, w),
            valid_ratio=data_dict.get('ocrreg_valid_ratio', 1)
        )
        data_sample.set_metainfo(img_meta)

        data_dict['data_samples'] = data_sample

        return data_dict
