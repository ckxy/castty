import os
import numpy as np
from PIL import Image
from .reader import Reader
from .builder import READER
from ..utils.structures import Meta
from ..utils.common import get_image_size

from text_renderer.render import Render
from text_renderer.config import get_cfg


__all__ = ['TextGenReader']


@READER.register_module()
class TextGenReader(Reader):
    def __init__(self, path, **kwargs):
        super(TextGenReader, self).__init__(**kwargs)

        assert os.path.exists(path)

        self.path = path

        generator_cfg = get_cfg(path)[0]
        self.render = Render(generator_cfg.render_cfg)
        self.num_images = generator_cfg.num_image

        self._info = dict(
            forcat=dict(
                type='seq',
            )
        )

    def __call__(self, index):
        data = self.render()

        img = Image.fromarray(data[0]).convert('RGB')
        if not self.use_pil:
            img = np.asarray(img)

        w, h = get_image_size(img)

        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=f'{self.path}--{index}',
            seq=data[1],
        )

    def __len__(self):
        return self.num_images

    def __repr__(self):
        return 'TextGenReader(path={}, {})'.format(self.path, super(TextGenReader, self).__repr__())
