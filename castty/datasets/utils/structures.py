import numpy as np


class Meta(dict):
    filter_flag = False
    
    def __init__(self, *args, **kwargs):
        tmp = None
        for k, v in kwargs.items():
            if not isinstance(v, np.ndarray):
                raise ValueError(f"'{k}' is not a ndarray")

            if tmp is not None:
                if len(tmp) != len(v):
                    raise ValueError('different length')
            tmp = v

        super(Meta, self).__init__(*args, **kwargs)
        # self.filter_flag = False

    # def get(self, key):
    #     return super(Meta, self).__getitem__(key)

    # def __getitem__(self, key):
    #     if key in self.keys():
    #         return super(Meta, self).__getitem__(key)
    #     else:
    #         return None
    
    def __setitem__(self, key, value):
        if not self.filter_flag:
            if not isinstance(value, np.ndarray):
                raise ValueError(f"'{key}' is not a ndarray")

            if len(self.keys()) > 0 and len(self[list(self.keys())[0]]) != len(value):
                raise ValueError('incorrect length')

        super(Meta, self).__setitem__(key, value)

    def filter(self, keep):
        # if not isinstance(keep, list) or (len(keep) > 0 and not isinstance(keep[0], int)):
        #     raise ValueError('illegal list')
        if not isinstance(keep, np.ndarray) or keep.dtype != bool:
            raise ValueError('illegal bool ndarray')

        self.filter_flag = True
        for key in self.keys():
            self[key] = self[key][keep]
        self.filter_flag = False

    def __add__(self, other):
        if self.keys() != other.keys():
            raise KeyError('different keys')

        m = Meta()
        for key in self.keys():
            v = np.concatenate([self[key], other[key]])
            m[key] = v
        return m

if __name__ == '__main__':
    # m = Meta(b=2)
    m = Meta(name=np.array([1, 0, 1, 6, 8]), class_id=np.array([1, 0, 1, 0, 0]))
    print(m['name'])
    # print(m['c'])
    # m['name'] = 0
    m['name'] = np.array([9, 8, 0, 5, 4])
    m['flag'] = np.array([True, False, True, False, False])

    m.filter([0, 1, 4])
    print(m, m.keys())

    n = Meta(
        name=np.array([1, 0, 1, 6, 8]), 
        class_id=np.array([2, 2, 1, 2, 0]),
        flag=np.array([False, False, True, True, False])
    )
    print(n, n.keys())

    m += n
    print(m, m.keys())
