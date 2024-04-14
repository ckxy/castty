import os
import cv2
import numpy as np
from PIL import Image
from .reader import Reader
from .builder import READER
from .utils import read_image_paths
from ..utils.structures import Meta
from ..utils.common import get_image_size


__all__ = ['CanvasLayoutReader']


@READER.register_module()
class CanvasLayoutReader(Reader):
    def __init__(self, root, csv_path, **kwargs):
        super(CanvasLayoutReader, self).__init__(**kwargs)

        assert os.path.exists(os.path.join(root, 'inpainted_poster'))
        assert os.path.exists(os.path.join(root, 'saliencymaps_basnet'))
        assert os.path.exists(os.path.join(root, 'saliencymaps_pfpn'))
        assert os.path.exists(csv_path)

        self.root = root
        self.csv_path = csv_path

        self.image_paths = read_image_paths(os.path.join(root, 'inpainted_poster'))
        self.sm_bas_paths = read_image_paths(os.path.join(root, 'saliencymaps_basnet'))
        self.sm_pfpn_paths = read_image_paths(os.path.join(root, 'saliencymaps_pfpn'))

        assert len(self.image_paths) > 0
        assert len(self.image_paths) == len(self.sm_bas_paths)
        assert len(self.image_paths) == len(self.sm_pfpn_paths)

        from pandas import read_csv

        df = read_csv(csv_path)
        self.groups = df.groupby(df.poster_path)

        # bboxes = df.box_elem.values.tolist()
        # bboxes = [eval(b) for b in bboxes]
        # bboxes = np.array(bboxes).astype(np.float32)
        # print(bboxes, bboxes.shape)
        # print(np.max(bboxes, axis=0))
        # print(np.min(bboxes, axis=0))
        # exit()

        self.classes = ['text', 'logo', 'underlay']

        self._info = dict(
            forcat=dict(
                bbox=dict(
                    classes=self.classes,
                    extra_meta=['difficult']
                )
            ), 
            tag_mapping=dict(
                image=['image', 'saliency_map'],
                bbox=['bbox']
            )
        )

    def __getitem__(self, index):
        image = self.read_image(self.image_paths[index])
        saliency_map_basnet = self.read_image(self.sm_bas_paths[index])
        saliency_map_pfpn = self.read_image(self.sm_pfpn_paths[index])

        if self.use_pil:
            saliency_map_basnet = saliency_map_basnet.convert('L')
            saliency_map_pfpn = saliency_map_pfpn.convert('L')
            saliency_map = Image.fromarray(np.maximum(np.array(saliency_map_basnet), np.array(saliency_map_pfpn)))
            saliency_map = saliency_map.convert('RGB')
        else:
            saliency_map_basnet = cv2.cvtColor(saliency_map_basnet, cv2.COLOR_RGB2GRAY)
            saliency_map_pfpn = cv2.cvtColor(saliency_map_pfpn, cv2.COLOR_RGB2GRAY)
            saliency_map = np.maximum(saliency_map_basnet, saliency_map_pfpn)
            saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_GRAY2RGB)

        w, h = get_image_size(image)

        poster_name = os.path.basename(self.image_paths[index]).replace("_mask", "")
        poster_name = os.path.join('train', poster_name)

        sliced_df = self.groups.get_group(poster_name)

        classes = np.array(list(sliced_df["cls_elem"])).astype(np.int32) - 1
        bboxes = np.array(list(map(eval, sliced_df["box_elem"]))).astype(np.float32)

        bbox_meta = Meta(
            class_id=classes,
            score=np.ones(len(bboxes)).astype(np.float32),
            keep=np.ones(len(bboxes)).astype(np.bool_),
        )

        return dict(
            image=image,
            image_meta=dict(ori_size=(w, h), path=self.image_paths[index]),
            saliency_map=saliency_map,
            saliency_map_meta=dict(ori_size=(w, h), path=self.sm_bas_paths[index]),
            bbox=bboxes,
            bbox_meta=bbox_meta
        )

    def __len__(self):
        return len(self.image_paths)

    def __repr__(self):
        return f'CanvasLayoutReader(root={self.root}, csv_path={self.csv_path}, {super(CanvasLayoutReader, self).__repr__()})'
