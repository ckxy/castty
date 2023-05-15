# copy from https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/dataset.py

import os
import re
import six
import numpy as np
from PIL import Image
from .reader import Reader
from .builder import READER
from ..utils.structures import Meta
from ..utils.common import get_image_size


__all__ = ['LmdbDTRBReader']


@READER.register_module()
class LmdbDTRBReader(Reader):
    def __init__(self, root, char_path, max_length=25, data_filtering_off=False, sensitive=False, **kwargs):
        super(LmdbDTRBReader, self).__init__(**kwargs)

        assert os.path.exists(root)
        assert os.path.exists(char_path)
        assert max_length > 0

        self.root = root
        self.char_path = char_path
        self.max_length = max_length
        self.data_filtering_off = data_filtering_off
        self.sensitive = sensitive

        import lmdb
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            raise FileNotFoundError(f'cannot create lmdb from {root}')

        with open(char_path, 'r') as f:
            self.character = ''.join(f.readlines())

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

            if self.data_filtering_off:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                """ Filtering part
                If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                use --data_filtering_off and only evaluate on alphabets and digits.
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192
                And if you want to evaluate them with the model trained with --sensitive option,
                use --sensitive and --data_filtering_off,
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
                """
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')

                    if len(label) > self.max_length:
                        # print(f'The length of the label is longer than max_length: length
                        # {len(label)}, {label} in dataset {self.root}')
                        continue

                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    out_of_char = f'[^{self.character}]'
                    if re.search(out_of_char, label.lower()):
                        continue

                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)

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
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-{:0>9d}'.format(index).encode()
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-{:0>9d}'.format(index).encode()
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')  # for color image
            if not self.use_pil:
                img = np.array(img)

            if not self.sensitive:
                label = label.lower()

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            out_of_char = f'[^{self.character}]'
            label = re.sub(out_of_char, '', label)

        w, h = get_image_size(img)
        # label = '了呗的愤世嫉俗繁华似u发挥'

        return dict(
            image=img,
            image_meta=dict(ori_size=(w, h), path=f'{self.root}--{index}'),
            seq=label,
        )

    def __len__(self):
        return self.nSamples

    def __repr__(self):
        return 'LmdbDTRBReader(root={}, char_path={}, max_length={}, data_filtering_off={}, sensitive={}, {})'.format(self.root, self.char_path, self.max_length, self.data_filtering_off, self.sensitive, super(LmdbDTRBReader, self).__repr__())
