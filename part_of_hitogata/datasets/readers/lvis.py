import os
import numpy as np
from PIL import Image
from addict import Dict
from copy import deepcopy
from .reader import Reader
from .lvis_v1_categories import LVIS_CATEGORIES as LVIS_V1_CATEGORIES
from .builder import READER
from ..utils.structures import Meta
from ..utils.common import get_image_size


__all__ = ['LVISAPIReader']


@READER.register_module()
class LVISAPIReader(Reader):
    def __init__(self, set_path, img_root, **kwargs):
        super(LVISAPIReader, self).__init__(**kwargs)

        self.set = set_path
        self.img_root = img_root

        assert os.path.exists(self.set)
        assert os.path.exists(self.img_root)

        assert len(LVIS_V1_CATEGORIES) == 1203
        cat_ids = [k["id"] for k in LVIS_V1_CATEGORIES]
        assert min(cat_ids) == 1 and max(cat_ids) == len(
            cat_ids
        ), "Category ids are not in [1, #categories], as expected"
        # Ensure that the category list is sorted by id
        lvis_categories = sorted(LVIS_V1_CATEGORIES, key=lambda x: x["id"])
        self.thing_classes = [k["synonyms"][0] for k in lvis_categories]
        self.meta = dict(thing_classes=self.thing_classes)

        self.data_lines = self.load_lvis_json()

        self._info = dict(
            forcat=dict(
                type='det',
                bbox_classes=self.thing_classes
            ),
        )

    def load_lvis_json(self):
        """
        Load a json file in LVIS's annotation format.
        Args:
            extra_annotation_keys (list[str]): list of per-annotation keys that should also be
                loaded into the dataset dict (besides "bbox", "bbox_mode", "category_id",
                "segmentation"). The values for these keys will be returned as-is.
        Returns:
            list[dict]: a list of dicts in Detectron2 standard format. (See
            `Using Custom Datasets </tutorials/datasets.html>`_ )
        Notes:
            1. This function does not read the image files.
               The results do not have the "image" field.
        """
        from lvis import LVIS

        lvis_api = LVIS(self.set)

        # sort indices for reproducible results
        img_ids = sorted(lvis_api.imgs.keys())
        # imgs is a list of dicts, each looks something like:
        # {'license': 4,
        #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
        #  'file_name': 'COCO_val2014_000000001268.jpg',
        #  'height': 427,
        #  'width': 640,
        #  'date_captured': '2013-11-17 05:57:24',
        #  'id': 1268}
        imgs = lvis_api.load_imgs(img_ids)
        # anns is a list[list[dict]], where each dict is an annotation
        # record for an object. The inner list enumerates the objects in an image
        # and the outer list enumerates over images. Example of anns[0]:
        # [{'segmentation': [[192.81,
        #     247.09,
        #     ...
        #     219.03,
        #     249.06]],
        #   'area': 1035.749,
        #   'image_id': 1268,
        #   'bbox': [192.81, 224.8, 74.73, 33.43],
        #   'category_id': 16,
        #   'id': 42986},
        #  ...]
        anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

        # Sanity check that each annotation has a unique id
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique".format(
            json_file
        )

        imgs_anns = list(zip(imgs, anns))

        extra_annotation_keys = []

        def get_file_name(img_root, img_dict):
            # Determine the path including the split folder ("train2017", "val2017", "test2017") from
            # the coco_url field. Example:
            #   'coco_url': 'http://images.cocodataset.org/train2017/000000155379.jpg'
            split_folder, file_name = img_dict["coco_url"].split("/")[-2:]
            return os.path.join(img_root, split_folder, file_name)

        dataset_dicts = []

        for (img_dict, anno_dict_list) in imgs_anns:
            record = dict()
            record["path"] = get_file_name(self.img_root, img_dict)
            record['ori_size'] = np.array([img_dict["height"], img_dict["width"]]).astype(np.float32)
            record["not_exhaustive_category_ids"] = img_dict.get("not_exhaustive_category_ids", [])
            record["not_exhaustive_category_ids"] = [i - 1 for i in record["not_exhaustive_category_ids"]]
            record["neg_category_ids"] = img_dict.get("neg_category_ids", [])
            record["neg_category_ids"] = [i - 1 for i in record["neg_category_ids"]]
            image_id = record["id"] = img_dict["id"]

            boxes = []
            classes = []
            for anno in anno_dict_list:
                # Check that the image_id in this annotation is the same as
                # the image_id we're looking at.
                # This fails only when the data parsing logic or the annotation file is buggy.
                assert anno["image_id"] == image_id
                x1, y1, w, h = anno["bbox"]
                boxes.append([x1, y1, x1 + w, y1 + h])  # Convert 1-indexed to 0-indexed
                classes.append(anno["category_id"] - 1)

                # segm = anno["segmentation"]  # list[list[float]]
                # # filter out invalid polygons (< 3 points)
                # valid_segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                # assert len(segm) == len(
                #     valid_segm
                # ), "Annotation contains an invalid polygon with < 3 points"
                # assert len(segm) > 0
                # obj["segmentation"] = segm

            if len(boxes) == 0:
                record["bbox"] = np.zeros((0, 4)).astype(np.float32)
            else:
                record["bbox"] = np.array(boxes)
            record["class_id"] = np.array(classes).astype(np.int32)
            dataset_dicts.append(record)

        return dataset_dicts

    def __call__(self, index):
        data_line = deepcopy(self.data_lines[index])
        data_line['image'] = self.read_image(data_line['path'])
        data_line['bbox_meta'] = Meta(
            class_id=data_line.pop('class_id'),
            score=np.ones(len(data_line['bbox']))
        )
        # data_line['bbox'] = data_line['bbox'][..., :4]
        return data_line

    def __len__(self):
        return len(self.data_lines)

    def __repr__(self):
        return 'LVISAPIReader(set_path={}, img_root={}, {})'.format(self.set, self.img_root, super(LVISAPIReader, self).__repr__())
