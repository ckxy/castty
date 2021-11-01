import os
import ntpath
from addict import Dict
from copy import deepcopy
from PIL import Image
from .reader import Reader
from .utils import read_image_paths


__all__ = ['C2NReader', 'CMTETestReader']


class C2NReader(Reader):
    def __init__(self, root, preload=True, **kwargs):
        super(C2NReader, self).__init__(**kwargs)

        self.root = root
        self.preload = preload

        id_root = os.path.join(self.root, 'train')
        tr_root = os.path.join(self.root, '2n_trans')
        assert os.path.exists(id_root)
        assert os.path.exists(tr_root)

        id_paths = read_image_paths(id_root)
        names = [os.path.splitext(ntpath.basename(id_))[0] for id_ in id_paths]
        assert len(id_paths) == len(set(names))
        self.n = len(id_paths)

        self.l = []
        for i in range(len(id_paths)):
            self.l.append((i, i))
            self.l.append((-1, i))

        if self.preload:
            self.preload_data = []

            for ai, bi in self.l:
                if ai == bi:
                    self.preload_data.append(dict(
                        a_image=self.read_image(id_paths[ai]),
                        a_star_image=self.read_image(id_paths[ai]),
                        b_image=self.read_image(id_paths[bi]),
                        a_path=id_paths[ai],
                        a_star_path=id_paths[ai],
                        b_path=id_paths[bi],
                        a_label=ai,
                        b_label=bi
                    ))
                else:
                    b_path = read_image_paths(os.path.join(tr_root, names[bi]))[0]
                    a_label = names.index(os.path.splitext(os.path.basename(b_path))[0])
                    a_path = id_paths[a_label]

                    a_img = self.read_image(a_path)
                    b_img = self.read_image(b_path)
                    assert a_img.size == b_img.size

                    self.preload_data.append(dict(
                        a_image=a_img,
                        a_star_image=self.read_image(id_paths[bi]),
                        b_image=b_img,
                        a_path=a_path,
                        a_star_path=id_paths[bi],
                        b_path=b_path,
                        a_label=a_label,
                        b_label=bi
                    ))

        self.data_lines = [0] * len(self.l)
        assert len(self.data_lines) > 0

    def get_dataset_info(self):
        return range(len(self.data_lines)), Dict({})

    def get_data_info(self, index):
        return

    def __call__(self, index):
        # index = data_dict
        if self.preload:
            return deepcopy(self.preload_data[index % (2 * self.n)])
        else:
            ai, bi = self.l[index % (2 * self.n)]
            if ai == bi:
                return dict(
                    a_image=self.read_image(id_paths[ai]),
                    a_star_image=self.read_image(id_paths[ai]),
                    b_image=self.read_image(id_paths[bi]),
                    a_path=id_paths[ai],
                    a_star_path=id_paths[ai],
                    b_path=id_paths[bi],
                    a_label=ai,
                    b_label=bi
                )
            else:
                b_path = read_image_paths(os.path.join(tr_root, names[bi]))[0]
                a_label = names.index(os.path.splitext(os.path.basename(b_path))[0])
                a_path = id_paths[a_label]

                a_img = self.read_image(a_path)
                b_img = self.read_image(b_path)
                assert a_img.size == b_img.size

                return dict(
                    a_image=a_img,
                    a_star_image=self.read_image(id_paths[bi]),
                    b_image=b_img,
                    a_path=a_path,
                    a_star_path=id_paths[bi],
                    b_path=b_path,
                    a_label=a_label,
                    b_label=bi
                )

    def __repr__(self):
        return 'C2NReader(root={}, preload={}, n={}, {})'.format(self.root, self.preload, self.n, super(C2NReader, self).__repr__())


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
