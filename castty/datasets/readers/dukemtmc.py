import os
import numpy as np
from .reader import Reader
from .builder import READER


__all__ = ['DukeMTMCAttritubesReader']


@READER.register_module()
class DukeMTMCAttritubesReader(Reader):
    def __init__(self, root, group='train', mode='ab', **kwargs):
        super(DukeMTMCAttritubesReader, self).__init__(**kwargs)

        self.root = root
        classes = ('backpack', 'bag', 'handbag', 
            'boots', 'gender', 'hat', 'shoes', 
            'top', 'downblack', 'downwhite', 
            'downred', 'downgray', 'downblue', 
            'downgreen', 'downbrown', 'upblack', 
            'upwhite', 'upred', 'uppurple', 
            'upgray', 'upblue', 'upgreen', 'upbrown')

        if group == 'train':
            img_root = os.path.join(root , 'bounding_box_train')
            self.classes = classes
        elif group == 'test':
            img_root = os.path.join(root, 'bounding_box_test')
            self.classes = ('boots', 'shoes', 'top', 
                'gender', 'hat', 'backpack', 'bag', 
                'handbag', 'downblack', 'downwhite', 
                'downred', 'downgray', 'downblue', 
                'downgreen', 'downbrown', 'upblack', 
                'upwhite', 'upred', 'upgray', 'upblue', 
                'upgreen', 'uppurple', 'upbrown')
        else:
            raise ValueError

        self.mode = mode

        if self.mode == 'b':
            self.grouped_classes = (
                ('carrying backpack',),
                ('carrying bag',),
                ('carrying handbag',),
                ('wearing boots',),
                ('gender',),
                ('wearing hat',),
                ('color of shoes',),
                ('length of upper-body clothing',),
                ('downunknown', 'downblack', 'downwhite', 'downred', 'downgray', 'downblue', 'downgreen', 'downbrown'),
                ('upunknown', 'upblack', 'upwhite', 'upred', 'uppurple', 'upgray', 'upblue', 'upgreen', 'upbrown'),
            )
        elif self.mode == 'c':
            self.grouped_classes = (
                ('without backpack', 'with backpack'),
                ('without bag', 'with bag'),
                ('without handbag', 'with handbag'),
                ('without boots', 'with boots'),
                ('male', 'female'),
                ('without hat', 'with hat'),
                ('dark', 'light'),
                ('short upper body clothing', 'long upper body clothing'),
                ('downunknown', 'downblack', 'downwhite', 'downred', 'downgray', 'downblue', 'downgreen', 'downbrown'),
                ('upunknown', 'upblack', 'upwhite', 'upred', 'uppurple', 'upgray', 'upblue', 'upgreen', 'upbrown'),
            )
        elif self.mode == 'ab':
            self.grouped_classes = (
                ('carrying backpack',),
                ('carrying bag',),
                ('carrying handbag',),
                ('wearing boots',),
                ('gender',),
                ('wearing hat',),
                ('color of shoes',),
                ('length of upper-body clothing',),
                ('downblack',), 
                ('downwhite',), 
                ('downred',), 
                ('downgray',), 
                ('downblue',), 
                ('downgreen',), 
                ('downbrown',),
                ('upblack',), 
                ('upwhite',), 
                ('upred',), 
                ('uppurple',), 
                ('upgray',), 
                ('upblue',), 
                ('upgreen',), 
                ('upbrown',),
            )
        else:
            raise ValueError

        self.mapping = []
        for i, c in enumerate(classes):
            self.mapping.append(self.classes.index(c))

        self.group = group

        self.pids = []
        self.img_paths = []
        tmp = sorted(os.listdir(img_root))
        for t in tmp:
            self.img_paths.append(os.path.join(img_root, t))
            self.pids.append(t.split('_')[0])
        self.pids = sorted(list(set(self.pids)))

        from scipy.io import loadmat
        self.f = loadmat(os.path.join(root, 'DukeMTMC-attribute', 'duke_attribute.mat'))
        self.f = self.f['duke_attribute'][0][0][1 if group == 'test' else 0][0][0]

        assert len(self.img_paths) > 0

        self._info = dict(
            forcat=dict(
                label=dict(
                    classes=self.grouped_classes
                ),
            ),
        )

    def __getitem__(self, index):
        img = self.read_image(self.img_paths[index])
        w, h = img.size
        path = self.img_paths[index]

        pid = self.pids.index(os.path.splitext(os.path.basename(path))[0].split('_')[0])

        labels = []
        tmp = []

        if self.mode == 'ab':
            for i in range(len(self.classes)):
                labels.append(self.f[self.mapping[i]][0][pid] - 1)
        else:
            for i in range(len(self.classes)):
                if 8 <= i <= 14:
                    tmp.append(self.f[self.mapping[i]][0][pid])
                    if i == 14:
                        labels.append(tmp.index(2) + 1)
                        tmp.clear()
                elif 15 <= i <= 22:
                    tmp.append(self.f[self.mapping[i]][0][pid])
                    if i == 22:
                        labels.append(tmp.index(2) + 1)
                        tmp.clear()
                else:
                    labels.append(self.f[self.mapping[i]][0][pid] - 1)

        if -1 in labels:
            if pid == 165:
                labels[7] = 0
            elif pid == 326:
                labels[7] = 1

        grouped_labels = []
        for i, gc in enumerate(self.grouped_classes):
            # print(gc)
            if len(gc) == 1:
                gl = np.array([labels[i]]).astype(np.int32)
            else:
                gl = np.zeros(len(gc)).astype(np.int32)
                gl[labels[i]] = 1
            grouped_labels.append(gl)

        return dict(
            image=img,
            image_meta=dict(ori_size=(w, h), path=path),
            label=grouped_labels
        )

    def __len__(self):
        return len(self.img_paths)

    def __repr__(self):
        return 'DukeMTMCAttritubesReader(root={}, group={}, mode={}, {})'.format(self.root, self.group, self.mode, super(DukeMTMCAttritubesReader, self).__repr__())
