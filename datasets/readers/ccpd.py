import os
import numpy as np
from addict import Dict
from PIL import Image
from .reader import Reader
from .builder import READER


__all__ = ['CCPD2019FolderReader']


@READER.register_module()
class CCPD2019FolderReader(Reader):
    def __init__(self, root, **kwargs):
        super(CCPD2019FolderReader, self).__init__(**kwargs)

        self.root = root
        self.chars = ('京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
                     '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
                     '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
                     '新', 
                     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                     'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                     'W', 'X', 'Y', 'Z', 'I', 'O', '-')
        self.img_paths = sorted(os.listdir(kwargs['root']))
        assert len(self.img_paths) > 0

    def get_dataset_info(self):
        return range(len(self.img_paths)), Dict({'chars': self.chars})

    def get_data_info(self, index):
        img = Image.open(self.img_paths[index][0])
        w, h = img.size
        return dict(h=h, w=w)

    def __call__(self, index):
        # index = data_dict
        # img = Image.open(os.path.join(self.root, self.img_paths[index])).convert('RGB')
        img = self.read_image(os.path.join(self.root, self.img_paths[index]))
        w, h = img.size
        path = os.path.join(self.root, self.img_paths[index])

        base_name = os.path.basename(self.img_paths[index])
        img_name, suffix = os.path.splitext(base_name)
        img_name = img_name.split("-")[0].split("_")[0]

        # if len(img_name) == 8:
        #     print(path, 'a')
        #     if img_name[2] != 'D' and img_name[2] != 'F' and img_name[-1] != 'D' and img_name[-1] != 'F':
        #         print(path)
        #         raise ValueError

        words = []
        for c in img_name:
            words.append(self.chars.index(c))

        # return {'image': img, 'ori_size': np.array([h, w]).astype(np.float32), 'path': path, 'seq': words, 'seq_length': len(words)}
        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=path,
            seq=words,
            seq_length=len(words)
        )

    def __repr__(self):
        return 'CCPD2019FolderReader(root={}, {})'.format(self.root, super(CCPD2019FolderReader, self).__repr__())
