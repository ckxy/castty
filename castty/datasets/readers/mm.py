import os
import json
from .reader import Reader
from .builder import READER
from ..utils.common import get_image_size


__all__ = ['MMOCRRegReader']


@READER.register_module()
class MMOCRRegReader(Reader):
    def __init__(self, json_path, **kwargs):
        super(MMOCRRegReader, self).__init__(**kwargs)

        assert os.path.exists(json_path)
        self.json_path = json_path
        self.root = os.path.dirname(json_path)

        with open(json_path, 'r') as f:
            data = json.load(f)
            
            meta_info = data['metainfo']
            self.data_list = data['data_list']

        assert meta_info['task_name'] == 'textrecog'
        assert meta_info['dataset_type'] == 'TextRecogDataset'

        self._info = dict(
            forcat=dict(
                seq=dict(),
            ),
            tag_mapping=dict(
                image=['image'],
                seq=['seq'],
            )
        )

    def __getitem__(self, index):
        data = self.data_list[index]

        img = self.read_image(os.path.join(self.root, data['img_path']))
        text = data['instances'][0]['text']
        w, h = get_image_size(img)

        return dict(
            image=img,
            image_meta=dict(ori_size=(w, h), path=os.path.join(self.root, data['img_path'])),
            seq=text,
        )

    def __len__(self):
        return len(self.data_list)

    def __repr__(self):
        return 'MMOCRRegReader(json_path={}, {})'.format(self.json_path, super(MMOCRRegReader, self).__repr__())
