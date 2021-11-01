import os
import sys
import types
import ntpath
import importlib
from addict import Dict


def load_config(cfg_name):
    cfg_filename = 'configs.' + cfg_name
    mod = importlib.import_module(cfg_filename)
    cfg_dict = dict()
    for k, v in mod.__dict__.items():
        if not k.startswith('__') and not isinstance(v, types.ModuleType):
            cfg_dict[k] = v

    return Dict(cfg_dict)


def load_config_far_away(cfg_path):
    cfg_dir, cfg_name = os.path.split(cfg_path)
    sys.path.insert(0, cfg_dir)
    mod = importlib.import_module(cfg_name.replace('.py', ''))
    sys.path.pop(0)
    cfg_dict = dict()
    for k, v in mod.__dict__.items():
        if not k.startswith('__') and not isinstance(v, types.ModuleType):
            cfg_dict[k] = v

    return Dict(cfg_dict)
