import io
import os
import time
import torch
import pickle
import logging
from PIL import Image
# from Crypto import Random
# from Crypto.Cipher import AES
# from Crypto.Util.Padding import pad, unpad


def save(epoch, encryption, **kwargs):
    assert 'cfg' in kwargs.keys()
    cfg = kwargs['cfg']
    save_path = os.path.join(cfg.general.checkpoints_dir, cfg.general.experiment_name, '{:0>6d}'.format(epoch) + '.pth')
    torch.save(kwargs, save_path)

    if encryption is not None:
        data = torch.load(save_path, map_location=lambda storage, loc: storage)
        data = pickle.dumps(data)

        key = encryption['key']
        iv = encryption['iv']
        cipher = AES.new(key, AES.MODE_CBC, iv)

        encrypted_data = cipher.encrypt(pad(data, AES.block_size))

        with open(save_path, "wb") as f:
            f.write(encrypted_data)


def load(path, encryption):
    if encryption is None:
        return torch.load(path, map_location=lambda storage, loc: storage)
    else:
        with open(path, "rb") as f:
            encrypted_data = f.read()

        key = encryption['key']
        iv = encryption['iv']
        cipher = AES.new(key, AES.MODE_CBC, iv)

        p_data = unpad(cipher.decrypt(encrypted_data), AES.block_size)
        return pickle.loads(p_data)
        # return torch.load(io.BytesIO(p_data))


def get_encryption(path, name):
    if len(path) == 0:
        return None
    else:
        key = Random.get_random_bytes(32)
        iv = Random.new().read(AES.block_size)

        if not os.path.exists(os.path.join(path, name)):
            os.makedirs(os.path.join(path, name))
        else:
            raise ValueError

        with open(os.path.join(path, name, "key.bin"), "wb") as f:
            f.write(key)
        with open(os.path.join(path, name, "iv.bin"), "wb") as f:
            f.write(iv)

        return dict(
            key=key,
            iv=iv
        )


def load_encryption(path):
    if len(path) == 0:
        return None
    else:
        with open(os.path.join(path, "key.bin"), "rb") as f:
            key = f.read()
        with open(os.path.join(path, "iv.bin"), "rb") as f:
            iv = f.read()

        return dict(
            key=key,
            iv=iv
        )


def init_logger(cfg):
    new_fold(os.path.join(cfg.general.checkpoints_dir, cfg.general.experiment_name, 'log'))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s\t%(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    log_dir = os.path.join(cfg.general.checkpoints_dir, cfg.general.experiment_name, 'log',
                           time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    # log_dir = os.path.join(cfg.general.checkpoints_dir, cfg.general.experiment_name, 'log', 'log.log')

    fh = logging.FileHandler(log_dir)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # sh = logging.StreamHandler()
    # sh.setLevel(logging.INFO)
    # sh.setFormatter(formatter)
    # logger.addHandler(sh)


def new_folds(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            new_fold(path)
    else:
        new_fold(paths)


def new_fold(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_integer(m, n):
    assert m > 0
    assert n > 0
    assert m >= n

    quotient = int(m / n)
    remainder = m % n

    if remainder > 0:
        return [quotient] * (n - remainder) + [quotient + 1] * remainder
    if remainder < 0:
        return [quotient - 1] * -remainder + [quotient] * (n + remainder)

    return [quotient] * n


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
