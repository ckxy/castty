import copy
from collections import Iterable
from .base_internode import BaseInternode


__all__ = ['EraseTags', 'RenameTag', 'CopyTag']


class EraseTags(BaseInternode):
    def __init__(self, tags):
        if isinstance(tags, str):
            self.tags = [tags]
        else:
            if isinstance(tags, Iterable):
                self.tags = list(tags)
            else:
                raise ValueError

    def __call__(self, data_dict):
        # data_dict = super(EraseTags, self).__call__(data_dict)
        for tag in self.tags:
            data_dict.pop(tag)
        return data_dict

    def __repr__(self):
        return 'EraseTags(tags={})'.format(tuple(self.tags))

    def rper(self):
        return 'EraseTags(not available)'


class RenameTag(BaseInternode):
    def __init__(self, old_name, new_name):
        self.old_name = old_name
        self.new_name = new_name

    def __call__(self, data_dict):
        # data_dict = super(RenameTag, self).__call__(data_dict)
        data_dict[self.new_name] = data_dict.pop(self.old_name)
        return data_dict

    def reverse(self, **kwargs):
        if self.new_name in kwargs.keys():
            kwargs[self.old_name] = kwargs.pop(self.new_name)
        return kwargs

    def __repr__(self):
        return 'RenameTag(old_name={}, new_name={})'.format(self.old_name, self.new_name)

    def rper(self):
        return 'RenameTag(old_name={}, new_name={})'.format(self.new_name, self.old_name)


class CopyTag(BaseInternode):
    def __init__(self, src_tag, dst_tag):
        self.src_tag = src_tag
        self.dst_tag = dst_tag

    def __call__(self, data_dict):
        # data_dict = super(CopyTag, self).__call__(data_dict)
        data_dict[self.dst_tag] = copy.deepcopy(data_dict[self.src_tag])
        return data_dict

    def __repr__(self):
        return 'CopyTag(src_tag={}, dst_tag={})'.format(self.src_tag, self.dst_tag)

    def rper(self):
        return 'CopyTag(not available)'
