TAG_MAPPING = dict(
    image=['image'],
    label=['label'],
    bbox=['bbox'],
    mask=['mask'],
    point=['point'],
    poly=['poly'],
    seq=['seq']
)


def base_forward(data_dict):
    data_dict['bbox'] = 'low'
    return data_dict


FUNC_MAPPING = dict(
    image=base_forward,
    label=base_forward,
    bbox=base_forward,
    mask=base_forward,
    point=base_forward,
    poly=base_forward,
    seq=base_forward
)


class BaseInternode(object):
    def calc_intl_param_forward(self, data_dict):
        return data_dict

    def forward(self, data_dict):
        return data_dict

    def erase_intl_param_forward(self, data_dict):
        return data_dict

    def calc_intl_param_backward(self, data_dict):
        return data_dict

    def backward(self, data_dict):
        return data_dict

    def erase_intl_param_backward(self, data_dict):
        return data_dict

    def __call__(self, data_dict):
        data_dict = self.calc_intl_param_forward(data_dict)
        data_dict = self.forward(data_dict)
        data_dict = self.erase_intl_param_forward(data_dict)
        return data_dict

    def reverse(self, **kwargs):
        kwargs = self.calc_intl_param_backward(kwargs)
        kwargs = self.backward(kwargs)
        kwargs = self.erase_intl_param_backward(kwargs)
        return kwargs


class SpecificTagsMixin(object):
    tag_mapping = TAG_MAPPING
    func_mapping = FUNC_MAPPING

    def forward_specific_tags(self, data_dict):
        for map2func, tags in self.tag_mapping.items():
            for tag in tags:
                if tag in data_dict.keys():
                    data_dict['intl_base_target_tag'] = tag
                    # data_dict = getattr(self, 'forward_{}'.format(tag))(data_dict)
                    data_dict = self.func_mapping[tag](data_dict)
                    data_dict.pop('intl_base_target_tag')
        return data_dict


class PrintHAHA(BaseInternode, SpecificTagsMixin):
    def __init__(self):
        super(PrintHAHA, self).__init__()
        # self.func_mapping['bbox'] = self.forward_bbox

    def forward(self, data_dict):
        return self.forward_specific_tags(data_dict)

    def forward_bbox(self, data_dict):
        data_dict['bbox'] = 'haha'
        return data_dict

    def backward(self, data_dict):
        return data_dict


if __name__ == '__main__':
    # printhaha = PrintHAHA()
    # a = dict(bbox='c')
    # a = printhaha(a)
    # print(a)

    # from castty.datasets.bamboo import Bamboo

    # internodes = [
    #     dict(type='Bamboo', internodes=[dict(type='ToTensor')]),
    #     dict(type='ToTensor')
    # ]

    # b = Bamboo(internodes, tag_mapping='aaa')
    # print(b)

    from PIL import Image
    import numpy as np
    import torch
    from torchvision.transforms.functional import crop

    img = Image.open('images/test900.jpg').convert('RGB')
    print(img)
    img = np.array(img)
    img = torch.from_numpy(img)
    print(img.shape)
    img = img.permute(2, 0, 1)
    img = crop(img, 100, 500, 400, 700)
    img = img.permute(1, 2, 0)
    img = img.numpy()
    print(img.shape)
    # exit()
    img = Image.fromarray(img)
    # print(img)
    # exit()
    # img = img.crop(100, 500,)
    img.show()
