import os
import numpy as np
from .reader import Reader
from .builder import READER
from scipy.io import loadmat
from ..utils.structures import Meta
from ..utils.common import get_image_size


__all__ = ['MPIIReader']


@READER.register_module()
class MPIIReader(Reader):
    def __init__(self, root, set_path, **kwargs):
        super(MPIIReader, self).__init__(**kwargs)

        self.root = root
        self.img_root = os.path.join(self.root, 'images')
        
        assert os.path.exists(self.img_root)
        assert os.path.exists(set_path)

        self.set_path = set_path

        mat = loadmat(set_path)

        self.image_paths = []
        self.data_lines = []

        s = set()

        for i, (anno, train_flag) in enumerate(
            zip(mat['RELEASE']['annolist'][0, 0][0],
                mat['RELEASE']['img_train'][0, 0][0])):
            
            img_name = anno['image']['name'][0, 0][0]
 
            img_path = os.path.join(self.img_root, img_name)
            if not os.path.exists(img_path):
                continue

            if int(train_flag) != 1:
                continue

            # self.image_paths.append(img_path)
            
            # if 'x1' in str(anno['annorect'].dtype):
            #     head_rect = zip(
            #         [x1[0, 0] for x1 in anno['annorect']['x1'][0]],
            #         [y1[0, 0] for y1 in anno['annorect']['y1'][0]],
            #         [x2[0, 0] for x2 in anno['annorect']['x2'][0]],
            #         [y2[0, 0] for y2 in anno['annorect']['y2'][0]]
            #     )

            if 'annopoints' in str(anno['annorect'].dtype):
                # only one person
                annopoints = anno['annorect']['annopoints'][0]
                head_x1s = anno['annorect']['x1'][0]
                head_y1s = anno['annorect']['y1'][0]
                head_x2s = anno['annorect']['x2'][0]
                head_y2s = anno['annorect']['y2'][0]

                joint_poss = []
                vis_flags = []

                for annopoint, head_x1, head_y1, head_x2, head_y2 in zip(
                        annopoints, head_x1s, head_y1s, head_x2s, head_y2s):
                    if len(annopoint) > 0:
                        # head_rect = [float(head_x1[0, 0]),
                        #              float(head_y1[0, 0]),
                        #              float(head_x2[0, 0]),
                        #              float(head_y2[0, 0])]

                        # joint coordinates

                        joint_pos = -np.ones((1, 16, 2), dtype=np.float32)
                        vis_flag = np.zeros((1, 16)).astype(np.bool_)

                        annopoint = annopoint['point'][0, 0]
                        j_id = [j_i[0, 0] for j_i in annopoint['id'][0]]
                        x = [x[0, 0] for x in annopoint['x'][0]]
                        y = [y[0, 0] for y in annopoint['y'][0]]
                        # joint_pos = dict()

                        for _j_id, (_x, _y) in zip(j_id, zip(x, y)):
                            # joint_pos[str(_j_id)] = [float(_x), float(_y)]
                            joint_pos[0, _j_id, 0] = float(_x)
                            joint_pos[0, _j_id, 1] = float(_y)

                        if 'is_visible' in str(annopoint.dtype):
                            vis = [v[0] if v else [0]
                                   for v in annopoint['is_visible'][0]]
                            vis = dict([(k, int(v[0])) if len(v) > 0 else v
                                        for k, v in zip(j_id, vis)])
                            for k, v in vis.items():
                                if v == 1:
                                    vis_flag[0, k] = True

                        joint_poss.append(joint_pos)
                        vis_flags.append(vis_flag)

                if len(joint_poss) > 0:
                    joint_poss = np.concatenate(joint_poss, axis=0)
                    vis_flags = np.concatenate(vis_flags, axis=0)

                    self.data_lines.append((joint_poss, vis_flags))
                    self.image_paths.append(img_path)

        assert len(self.image_paths) > 0

        self._info = dict(
            forcat=dict(
                point=dict(classes=[str(i) for i in range(16)])
            )
        )

    def __getitem__(self, index):
        img_path = os.path.join(self.image_paths[index]) 
        img = self.read_image(img_path)
        w, h = get_image_size(img)

        joint_pos, vis_flags = self.data_lines[index]
        meta = Meta(keep=vis_flags)

        return dict(
            image=img,
            image_meta=dict(ori_size=(w, h), path=self.image_paths[index]),
            # ori_size=np.array([h, w]).astype(np.float32),
            # path=img_path,
            point=joint_pos,
            point_meta=meta
        )

    def __len__(self):
        return len(self.image_paths)

    def __repr__(self):
        return 'MPIIReader(root={}, set_path={}, {})'.format(self.root, self.set_path, super(MPIIReader, self).__repr__())
