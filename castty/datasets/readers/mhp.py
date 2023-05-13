import os
import numpy as np
from PIL import Image
from .reader import Reader
from .builder import READER
from ..utils.common import get_image_size


__all__ = ['MHPV1Reader']


CLASSES = ['__background__', 'hat', 'hair', 'sunglass', 'upper-clothes', 'skirt', 'pants', 'dress', 'belt' , 'left-shoe', 'right-shoe', 'face', 'left-leg', 'right-leg', 'left-arm' , 'right-arm', 'bag', 'scarf', 'torso-skin']


@READER.register_module()
class MHPV1Reader(Reader):
    def __init__(self, root, split, **kwargs):
        super(MHPV1Reader, self).__init__(**kwargs)

        self.root = root
        self.split = split

        img_root = os.path.join(self.root, 'images')
        mask_root = os.path.join(self.root, 'annotations')

        assert os.path.exists(img_root)
        assert os.path.exists(mask_root)

        id_list_file = os.path.join(self.root, '{}_list.txt'.format(self.split))

        self.image_paths = [os.path.join(img_root, id_.strip()) for id_ in open(id_list_file)]
        mask_paths = os.listdir(mask_root)

        self.name2mask = dict()
        for mask_path in mask_paths:
            items = mask_path.split('_')
            if items[0] in self.name2mask.keys():
                self.name2mask[items[0]].append(os.path.join(mask_root, mask_path))
            else:
                self.name2mask[items[0]] = [os.path.join(mask_root, mask_path)]

        assert len(self.image_paths) > 0

        self._info = dict(
            forcat=dict(
                mask=dict(
                    classes=CLASSES
                )
            ),
            tag_mapping=dict(
                image=['image'],
                mask=['mask']
            )
        )

    def __getitem__(self, index):
        img = self.read_image(self.image_paths[index])
        w, h = get_image_size(img)

        name = os.path.splitext(os.path.basename(self.image_paths[index]))[0]
        mask_all = np.zeros((h, w)).astype(np.int32)

        for mask_path in self.name2mask[name]:
            mask = Image.open(mask_path)
            mask = np.array(mask).astype(np.int32)

            m = mask > 0
            mask_all[m] = mask[m]

        return dict(
            image=img,
            image_meta=dict(ori_size=(w, h), path=self.image_paths[index]),
            mask=mask_all
        )

    def __len__(self):
        return len(self.image_paths)

    def __repr__(self):
        return 'MHPV1Reader(root={}, split={}, {})'.format(self.root, self.split, super(MHPV1Reader, self).__repr__())
