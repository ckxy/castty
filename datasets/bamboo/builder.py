from utils.registry import Registry, build_from_cfg

INTERNODE = Registry('internode')


def build_internode(cfg, **default_args):
    # print(INTERNODE.module_dict.keys())
    # exit()
    if 'one_way' in cfg.keys():
        if cfg['one_way'] == 'forward':
            tmp_cfg = dict(
                type='ForwardOnly',
                internode=cfg
            )
            return build_from_cfg(tmp_cfg, INTERNODE, default_args)
        elif cfg['one_way'] == 'backward':
            tmp_cfg = dict(
                type='BackwardOnly',
                internode=cfg
            )
            return build_from_cfg(tmp_cfg, INTERNODE, default_args)

    if 'p' in cfg.keys():
        tmp_cfg = dict(
            type='RandomWarpper',
            p=cfg['p'],
            internode=cfg
        )
        return build_from_cfg(tmp_cfg, INTERNODE, default_args)

    return build_from_cfg(cfg, INTERNODE, default_args)
