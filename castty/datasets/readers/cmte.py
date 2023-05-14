import os
import numpy as np
from copy import deepcopy
from .reader import Reader
from .builder import READER
from .utils import read_image_paths


__all__ = ['C2NReader', 'CMTETestReader']


@READER.register_module()
class C2NReader(Reader):
    def __init__(self, root, **kwargs):
        super(C2NReader, self).__init__(**kwargs)

        self.root = root

        id_root = os.path.join(self.root, 'train')
        tr_root = os.path.join(self.root, '2n_trans')
        assert os.path.exists(id_root)
        assert os.path.exists(tr_root)

        id_paths = read_image_paths(id_root)
        names = [os.path.splitext(os.path.basename(id_))[0] for id_ in id_paths]
        assert len(id_paths) == len(set(names))
        self.n = len(id_paths)

        l = []
        for i in range(len(id_paths)):
            l.append((i, i))
            l.append((-1, i))

        self.preload_data = []

        for ai, bi in l:
            if ai == bi:
                a_image = self.read_image(id_paths[ai])
                b_image = self.read_image(id_paths[bi])

                a_label = np.zeros(self.n).astype(np.int32)
                a_label[ai] = 1
                b_label = np.zeros(self.n).astype(np.int32)
                b_label[bi] = 1

                self.preload_data.append(dict(
                    image=a_image,
                    a_star_image=a_image,
                    b_image=b_image,
                    image_meta=dict(ori_size=a_image.size, path=id_paths[ai]),
                    a_star_image_meta=dict(ori_size=a_image.size, path=id_paths[ai]),
                    b_image_meta=dict(ori_size=b_image.size, path=id_paths[bi]),
                    label=[a_label],
                    b_label=[b_label]
                ))
            else:
                b_path = read_image_paths(os.path.join(tr_root, names[bi]))[0]
                ai = names.index(os.path.splitext(os.path.basename(b_path))[0])
                a_path = id_paths[ai]

                a_image = self.read_image(a_path)
                a_star_image = self.read_image(id_paths[bi])
                b_image = self.read_image(b_path)
                assert a_image.size == b_image.size

                a_label = np.zeros(self.n).astype(np.int32)
                a_label[ai] = 1
                b_label = np.zeros(self.n).astype(np.int32)
                b_label[bi] = 1

                self.preload_data.append(dict(
                    image=a_image,
                    a_star_image=a_star_image,
                    b_image=b_image,
                    image_meta=dict(ori_size=a_image.size, path=a_path),
                    a_star_image_meta=dict(ori_size=a_image.size, path=id_paths[bi]),
                    b_image_meta=dict(ori_size=b_image.size, path=b_path),
                    label=[a_label],
                    b_label=[b_label]
                ))

        self._info = dict(
            forcat=dict(
                label=dict(
                    classes=[str(i) for i in range(self.n)]
                ),
            ),
            tag_mapping=dict(
                image=['image', 'a_star_image', 'b_image'],
                label=['label', 'b_label']
            )
        )

    def __getitem__(self, index):
        index = 1
        return deepcopy(self.preload_data[index % (2 * self.n)])

    def __len__(self):
        return 2 * self.n

    def __repr__(self):
        return 'C2NReader(root={}, n={}, {})'.format(self.root, self.n, super(C2NReader, self).__repr__())


# @READER.register_module()
class CMTETestReader(Reader):
    def __init__(self, **kwargs):
        super(CMTETestReader, self).__init__(**kwargs)

        assert ('c_root' in kwargs.keys()) ^ ('c_path' in kwargs.keys())
        assert ('s_root' in kwargs.keys()) ^ ('s_path' in kwargs.keys())

        self.c_paths = []
        self.s_paths = []

        if 'c_root' in kwargs.keys():
            assert os.path.exists(kwargs['c_root'])
            self.c_paths += read_image_paths(kwargs['c_root'])
        else:
            assert os.path.exists(kwargs['c_path'])
            self.c_paths.append(kwargs['c_path'])

        if 's_root' in kwargs.keys():
            assert os.path.exists(kwargs['s_root'])
            self.s_paths += read_image_paths(kwargs['s_root'])
        else:
            assert os.path.exists(kwargs['s_path'])
            self.s_paths.append(kwargs['s_path'])

        self.data_lines = [0] * len(self.s_paths) * len(self.c_paths)
        assert len(self.data_lines) > 0

    def get_dataset_info(self):
        return range(len(self.data_lines)), Dict({})

    def get_data_info(self, index):
        return

    def __call__(self, index):
        # index = data_dict
        c_id = index // len(self.s_paths)
        s_id = index % len(self.s_paths)

        return dict(
            a_image=self.read_image(self.c_paths[c_id]),
            a_star_image=self.read_image(self.s_paths[s_id]),
            a_path=self.c_paths[c_id],
            a_star_path=self.s_paths[s_id],
        )

    def __repr__(self):
        return 'CMTETestReader(n_style={}, n_content={}, {})'.format(len(self.s_paths), len(self.c_paths), super(CMTETestReader, self).__repr__())
