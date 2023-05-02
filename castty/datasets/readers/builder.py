from ...utils.registry import Registry, build_from_cfg

READER = Registry('reader')


def build_reader(cfg, **default_args):
    return build_from_cfg(cfg, READER, default_args)
