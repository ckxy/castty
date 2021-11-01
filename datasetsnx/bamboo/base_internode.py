class BaseInternode(object):
    def __call__(self, data_dict):
        return data_dict

    def reverse(self, **kwargs):
        return kwargs

    def __repr__(self):
        return 'BaseInternode()'

    def rper(self):
        return 'BaseInternode()'
