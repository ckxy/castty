import os
import json
import numpy as np
from .reader import Reader
from .builder import READER
from scipy.io import loadmat


__all__ = ['MPIIReader']


@READER.register_module()
class MPIIReader(Reader):
    def __init__(self, root, set_path, length=200, **kwargs):
        super(MPIIReader, self).__init__(**kwargs)

        self.root = root
        self.img_root = os.path.join(self.root, 'images')
        
        assert os.path.exists(self.img_root)
        assert os.path.exists(set_path)

        self.set_path = set_path
        self.length = length

        self.mat = loadmat(set_path, struct_as_record=False)['RELEASE'][0, 0]

        # for index in range(10):
        #     annotation_mpii = self.mat.__dict__['annolist'][0, index]
        #     img_name = annotation_mpii.__dict__['image'][0, 0].__dict__['name'][0]
        #     print(img_name)
        # exit()

        assert self.mat.__dict__['annolist'][0].shape[0] > 0

        self._info = dict(
            forcat=dict(
                point=dict()
            )
        )

    def __call__(self, index):
        annotation_mpii = self.mat.__dict__['annolist'][0, index]
        img_name = annotation_mpii.__dict__['image'][0, 0].__dict__['name'][0]

        img_path = os.path.join(self.img_root, img_name) 
        img = self.read_image(img_path)

        person_id = self.mat.__dict__['single_person'][index][0].flatten()
        print(person_id - 1)
        exit()
        for person in (person_id - 1):
            print(annotation_mpii.__dict__.keys())
            print(annotation_mpii.__dict__['annorect'][0, 0].__dict__['objpos'][0, 0].__dict__)
            exit()
            try:
                annopoints_img_mpii = annotation_mpii.__dict__['annorect'][0, person].__dict__['annopoints'][0, 0]
                num_joints = annopoints_img_mpii.__dict__['point'][0].shape[0]

                print(num_joints)

                # Iterate over present joints
                for i in range(num_joints):
                    x = annopoints_img_mpii.__dict__['point'][0, i].__dict__['x'].flatten()[0]
                    y = annopoints_img_mpii.__dict__['point'][0, i].__dict__['y'].flatten()[0]
                    id_ = annopoints_img_mpii.__dict__['point'][0, i].__dict__['id'][0][0]
                    vis = annopoints_img_mpii.__dict__['point'][0, i].__dict__['is_visible'].flatten()

                    print(x, y, id_, vis)

                    # No entry corresponding to visible
                    # if vis.size == 0:
                    #     vis = 1
                    # else:
                    #     vis = vis.item()

                    # gt_per_joint = np.array([x, y, vis]).astype(np.float16)
                    # gt_per_image[mpii_idx_to_jnt[id_]].append(gt_per_joint)

                annotated_person_flag = True
            except KeyError:
                # Person 'x' could not have annotated joints, hence move to person 'y'
                print('e')
                continue
        exit()
        
        res['point'] = self.h5f['part'][index] - 1
        res['visible'] = self.h5f['visible'][index].astype(np.int32)

        print(self.h5f['part'][index] - 1)
        print(self.h5f['visible'][index].astype(np.int32))
        exit()

        s = self.h5f['scale'][index]
        c = self.h5f['center'][index] - 1
        l = round(s * self.length / 2)
        x1, y1 = c[0] - l, c[1] - l
        x2, y2 = c[0] + l, c[1] + l
        res['bbox'] = np.array([[x1, y1, x2, y2]]).astype(np.float32)
        res['mpii_scale'] = s

        res['image'] = img
        res['path'] = img_path
        w, h = get_image_size(img)
        res['mpii_length'] = self.length

        return dict(
            image=img,
            ori_size=np.array([h, w]).astype(np.float32),
            path=img_path,
            point=np.array(a['joint_self'])[..., :2][np.newaxis, ...].astype(np.float32),
            point_meta=meta
        )

    def __len__(self):
        return self.mat.__dict__['annolist'][0].shape[0]

    def __repr__(self):
        return 'MPIIReader(root={}, set_path={}, length={}, {})'.format(self.root, self.set_path, self.length, super(MPIIReader, self).__repr__())
