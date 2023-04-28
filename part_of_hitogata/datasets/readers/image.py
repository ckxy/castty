import os
from .reader import Reader
from .builder import READER
from .utils import read_image_paths
from ..utils.common import get_image_size


__all__ = ['ImageReader']


@READER.register_module()
class ImageReader(Reader):
    def __init__(self, root, **kwargs):
        super(ImageReader, self).__init__(**kwargs)

        assert os.path.exists(root)
        self.root = root
        self.image_paths = read_image_paths(self.root)
        assert len(self.image_paths) > 0

        self._info = dict(forcat=dict())

    def __getitem__(self, index):
        img = self.read_image(self.image_paths[index])
        w, h = get_image_size(img)

        return dict(
            image=img,
            image_meta=dict(ori_size=(w, h), path=self.image_paths[index])
            # ori_size=np.array([h, w]).astype(np.float32),
            # path=self.image_paths[index]
        )

    def __len__(self):
        return len(self.image_paths)

    def __repr__(self):
        return 'ImageReader(root={}, {})'.format(self.root, super(ImageReader, self).__repr__())
