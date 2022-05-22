class BaseInternode(object):
    def __call__(self, data_dict):
        return data_dict

    def reverse(self, **kwargs):
        return kwargs

    def __repr__(self):
        return type(self).__name__ + '()'

    def rper(self):
        return type(self).__name__ + '(not available)'

    @staticmethod
    def cv2_backend(**kwargs):
        raise NotImplementedError

    @staticmethod
    def pil_backend(**kwargs):
        raise NotImplementedError

    @staticmethod
    def bbox_backend(**kwargs):
        raise NotImplementedError

    @staticmethod
    def point_backend(**kwargs):
        raise NotImplementedError

    @staticmethod
    def mask_backend(**kwargs):
        raise NotImplementedError

    @staticmethod
    def label_backend(**kwargs):
        raise NotImplementedError
