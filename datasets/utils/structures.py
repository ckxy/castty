import numpy as np


class Meta(object):
    def __init__(self, keys, values):
        # self.check(keys, values)

        self.keys = keys
        values = np.array(values)
        self.values = values.reshape(len(keys), -1).astype(np.float32)

    # @staticmethod
    # def check(keys, values):
    #     from collections.abc import Iterable
    #     assert isinstance(keys, Iterable)
    #     assert isinstance(values, np.ndarray)

    #     if len(keys) <= 0:
    #         assert len(keys) > 0
    #     elif len(keys) == 1:
    #         assert values.ndim == 1
    #     else:
    #         assert len(keys) == len(values)
        
    def append(self, keys, values):
        # self.check(keys, values)

        self.keys.extend(list(keys))
        values = np.array(values)
        values = values.reshape(len(keys), -1)
        self.values = np.vstack((self.values, values))

    def filter(self, keep):
        self.values = self.values[..., keep]

    def index(self, key):
        return self.keys.index(key)

    def get(self, key):
        return self.values[self.index(key)]

    def __add__(self, other):
        assert self.keys == other.keys
        values = np.concatenate([self.values, other.values], axis=-1)
        return Meta(self.keys, values)

    def __repr__(self):
        return '{}\n{}'.format(self.keys, self.values)


if __name__ == '__main__':
    m = Meta(['name'], np.array([1, 0, 1, 6, 8]))
    m.append(['key'], np.array([2, 0, 1, 9, 7]))
    print(m.keys)
    print(m.values, m.values.shape)
    m.filter([2, 3, 4])
    print(m.values, m.values.shape)
    print(m.index('key'))

    n = Meta(['name'], np.array([20, 21, 22, 23]))
    n.append(['key'], np.array([10, 11, 12, 13]))
    print(n)

    o = m + n
    print(o)
