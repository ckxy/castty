from ..utils import TAG_MAPPING
from .utils import read_image_pil, read_image_cv2


class Reader(object):
    def __init__(self, **kwargs):
        if 'use_pil' in kwargs.keys():
            self.use_pil = kwargs['use_pil']
        else:
            self.use_pil = True

        self._info = dict(tag_mapping=TAG_MAPPING)

    @property
    def tag_mapping(self):
        return self._tag_mapping

    @property
    def info(self):
        return self._info

    def __getitem__(self, index):
        raise NotImplementedError

    def __repr__(self):
        return 'use_pil={}'.format(self.use_pil)

    def read_image(self, path):
        if self.use_pil:
            return read_image_pil(path)
        else:
            return read_image_cv2(path)
