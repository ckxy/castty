import os
import ntpath
import numpy as np
from addict import Dict
from PIL import Image
from .reader import Reader
from .builder import READER


__all__ = ['Market1501AttritubesReader']


@READER.register_module()
class Market1501AttritubesReader(Reader):
    def __init__(self, root, group='train', mode='ab', **kwargs):
        super(Market1501AttritubesReader, self).__init__(**kwargs)

        self.root = root
        labels = ('age', 'backpack', 
            'bag', 'handbag', 
            'downblack', 'downblue', 
            'downbrown', 'downgray', 
            'downgreen', 'downpink', 
            'downpurple', 'downwhite', 
            'downyellow', 'upblack', 
            'upblue', 'upgreen', 'upgray', 
            'uppurple', 'upred', 'upwhite', 
            'upyellow', 'clothes', 'down', 
            'up', 'hair', 'hat', 'gender')

        if group == 'train':
            img_root = os.path.join(root , 'bounding_box_train')
            self.labels = labels
        elif group == 'test':
            img_root = os.path.join(root, 'bounding_box_test')
            self.labels = ('age', 'backpack', 
                'bag', 'handbag', 'clothes', 
                'down', 'up', 'hair', 'hat', 
                'gender', 'upblack', 
                'upwhite', 'upred', 
                'uppurple', 'upyellow', 
                'upgray', 'upblue', 
                'upgreen', 'downblack', 
                'downwhite', 'downpink', 
                'downpurple', 'downyellow', 
                'downgray', 'downblue', 
                'downgreen', 'downbrown')
        else:
            raise ValueError

        self.mode = mode

        if self.mode == 'b':
            self.grouped_labels = (
                ('young', 'teenager', 'adult', 'old'),
                ('carrying backpack',),
                ('carrying bag',),
                ('carrying handbag',),
                ('downunknown', 'downblack', 'downblue', 'downbrown', 'downgray', 'downgreen', 'downpink', 'downpurple', 'downwhite', 'downyellow'),
                ('upunknown', 'upblack', 'upblue', 'upgreen', 'upgray', 'uppurple', 'upred', 'upwhite', 'upyellow'),
                ('clothes',),
                ('length of lower-body clothing',),
                ('sleeve length',),
                ('hair length',),
                ('wearing hat',),
                ('gender',),
            )
        elif self.mode == 'c':
            self.grouped_labels = (
                ('young', 'teenager', 'adult', 'old'),
                ('without backpack', 'with backpack'),
                ('without bag', 'with bag'),
                ('without handbag', 'with handbag'),
                ('downunknown', 'downblack', 'downblue', 'downbrown', 'downgray', 'downgreen', 'downpink', 'downpurple', 'downwhite', 'downyellow'),
                ('upunknown', 'upblack', 'upblue', 'upgreen', 'upgray', 'uppurple', 'upred', 'upwhite', 'upyellow'),
                ('dress', 'pants'),
                ('long lower body clothing', 'short lower body clothing'),
                ('long sleeve', 'short sleeve'),
                ('short hair', 'long hair'),
                ('without hat', 'with hat'),
                ('male', 'female'),
            )
        elif self.mode == 'ab':
            # self.grouped_labels = []
            # for l in labels:
            #     self.grouped_labels.append((l,))
            # self.grouped_labels = tuple(self.grouped_labels)
            self.grouped_labels = (
                ('young',), 
                ('teenager',), 
                ('adult',),
                ('old',),
                ('carrying backpack',),
                ('carrying bag',),
                ('carrying handbag',),
                ('downblack',), 
                ('downblue',), 
                ('downbrown',), 
                ('downgray',),
                ('downgreen',),
                ('downpink',),
                ('downpurple',),
                ('downwhite',), 
                ('downyellow',),
                ('upblack',), 
                ('upblue',), 
                ('upgreen',), 
                ('upgray',), 
                ('uppurple',), 
                ('upred',), 
                ('upwhite',), 
                ('upyellow',),
                ('clothes',),
                ('length of lower-body clothing',),
                ('sleeve length',),
                ('hair length',),
                ('wearing hat',),
                ('gender',),
            )
        else:
            raise ValueError

        self.mapping = []
        for i, label in enumerate(labels):
            self.mapping.append(self.labels.index(labels[i]))

        self.group = group

        self.pids = []
        self.img_paths = []
        tmp = sorted(os.listdir(img_root))
        for t in tmp:
            if t.startswith('0000') or t.startswith('-1'):
                # print(t)
                continue
            self.img_paths.append(os.path.join(img_root, t))
            self.pids.append(t.split('_')[0])
        self.pids = sorted(list(set(self.pids)))

        from scipy.io import loadmat
        self.f = loadmat(os.path.join(root, 'attribute', 'market_attribute.mat'))
        self.f = self.f['market_attribute'][0][0][0 if group == 'test' else 1][0][0]

        assert len(self.img_paths) > 0

    def get_dataset_info(self):
        return range(len(self.img_paths)), Dict({'classes': self.grouped_labels})

    def get_data_info(self, index):
        img = Image.open(self.img_paths[index][0])
        w, h = img.size
        return dict(h=h, w=w)

    def __call__(self, index):
        # index = data_dict
        # index = 5605
        # img = Image.open(self.img_paths[index]).convert('RGB')
        img = self.read_image(self.img_paths[index])
        w, h = img.size
        path = self.img_paths[index]

        pid = self.pids.index(os.path.splitext(ntpath.basename(path))[0].split('_')[0])

        # from tqdm import tqdm
        # tqdm.write('{} {}'.format(index, path))

        labels = []
        tmp = []
        if self.mode == 'ab':
            for i in range(len(self.labels)):
                if i == 1:
                    l = [0, 0, 0, 0]
                    l[self.f[self.mapping[i]][0][pid] - 1] = 1
                    labels = l + labels
                else:
                    labels.append(self.f[self.mapping[i]][0][pid] - 1)
        else:
            for i in range(len(self.labels)):
                # print(i, self.labels[self.mapping[i]], self.f[self.mapping[i]][0][pid] - 1)
                if 4 <= i <= 12:
                    tmp.append(self.f[self.mapping[i]][0][pid])
                    if i == 12:
                        # print(tmp)
                        try:
                            labels.append(tmp.index(2) + 1)
                        except ValueError:
                            labels.append(0)
                        tmp.clear()
                elif 13 <= i <= 20:
                    tmp.append(self.f[self.mapping[i]][0][pid])
                    if i == 20:
                        # print(tmp)
                        try:
                            labels.append(tmp.index(2) + 1)
                        except ValueError:
                            labels.append(0)
                        tmp.clear()
                else:
                    labels.append(self.f[self.mapping[i]][0][pid] - 1)
                # labels.append(self.f[self.mapping[i]][0][pid] - 1)

        labels = np.array(labels).astype(np.long)
        # print(labels, len(labels))
        # exit()

        # return {'image': img, 'ori_size': np.array([h, w]).astype(np.float32), 'path': path, 'label': labels}
        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=path,
            label=labels
        )

    def __repr__(self):
        return 'Market1501AttritubesReader(root={}, group={}, mode={}, {})'.format(self.root, self.group, self.mode, super(Market1501AttritubesReader, self).__repr__())
