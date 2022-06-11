import numpy as np


class Meta(object):
    def __init__(self, keys, values):
        assert isinstance(keys, list) and isinstance(values, list)
        assert isinstance(values[0], np.ndarray)
        assert len(keys) == len(values)

        self.keys = keys
        self.values = values
        
    def append(self, keys, values):
        assert isinstance(keys, list) and isinstance(values, list)
        assert isinstance(values[0], np.ndarray)
        assert len(keys) == len(values)

        self.keys.extend(keys)
        self.values.extend(values)

    def filter(self, keep):
        for i in range(len(self.values)):
            self.values[i] = self.values[i][keep]

    def index(self, key):
        return self.keys.index(key)

    def get(self, key):
        return self.values[self.index(key)]

    def __add__(self, other):
        assert self.keys == other.keys
        # values = np.concatenate([self.values, other.values], axis=-1)
        values = []
        for v1, v2 in zip(self.values, other.values):
            values.append(np.concatenate([v1, v2]))
        return Meta(self.keys, values)

    def __repr__(self):
        s = 'Meta(\n'
        for key, value in zip(self.keys, self.values):
            s += f'  {key}: {value}\n'
        return s + ')'


if __name__ == '__main__':
    m = Meta(['name'], [np.array([1, 0, 1, 6, 8])])
    m.append(['key'], [np.array(['aa', 'bb', 'cc', 'dd', 'ee'])])
    print(m)
    m.filter([2, 3, 4])
    print(m)
    print(m.index('key'))

    n = Meta(['name'], [np.array([20, 21, 22, 23])])
    n.append(['key'], [np.array(['ww', 'xx', 'yy', 'zz'])])
    print(n)

    o = m + n
    print(o)

