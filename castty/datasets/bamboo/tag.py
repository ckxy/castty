import copy
from .builder import INTERNODE
from collections import Iterable
from .base_internode import BaseInternode


__all__ = ['EraseTags', 'RenameTag', 'CopyTag']


@INTERNODE.register_module()
class EraseTags(BaseInternode):
    def __init__(self, tags, **kwargs):
        if isinstance(tags, str):
            assert not tags.startswith('intl_')

            self.tags = [tags]
        else:
            if isinstance(tags, Iterable):
                for tag in tags:
                    assert not tag.startswith('intl_')

                self.tags = list(tags)
            else:
                raise ValueError

        BaseInternode.__init__(self, **kwargs)

    def forward(self, data_dict, **kwargs):
        for tag in self.tags:
            data_dict.pop(tag)
        return data_dict

    def __repr__(self):
        return 'EraseTags(tags={})'.format(tuple(self.tags))


@INTERNODE.register_module()
class RenameTag(BaseInternode):
    def __init__(self, old_name, new_name, **kwargs):
        assert not old_name.startswith('intl_')
        assert not new_name.startswith('intl_')

        self.old_name = old_name
        self.new_name = new_name

        BaseInternode.__init__(self, **kwargs)

    def forward(self, data_dict, **kwargs):
        data_dict[self.new_name] = data_dict.pop(self.old_name)
        return data_dict

    def backward(self, kwargs):
        if self.new_name in kwargs.keys():
            kwargs[self.old_name] = kwargs.pop(self.new_name)
        return kwargs

    def __repr__(self):
        return 'RenameTag(old_name={}, new_name={})'.format(self.old_name, self.new_name)

    def rper(self):
        return 'RenameTag(old_name={}, new_name={})'.format(self.new_name, self.old_name)


@INTERNODE.register_module()
class CopyTag(BaseInternode):
    def __init__(self, src_tag, dst_tag, **kwargs):
        assert not src_tag.startswith('intl_')
        assert not dst_tag.startswith('intl_')

        self.src_tag = src_tag
        self.dst_tag = dst_tag

        BaseInternode.__init__(self, **kwargs)

    def forward(self, data_dict, **kwargs):
        data_dict[self.dst_tag] = copy.deepcopy(data_dict[self.src_tag])
        return data_dict

    def __repr__(self):
        return 'CopyTag(src_tag={}, dst_tag={})'.format(self.src_tag, self.dst_tag)
