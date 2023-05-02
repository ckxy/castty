import torch
import numpy as np

from castty.configs import load_config, load_config_far_away
from castty.visualization.toto import Toto


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    cfg = load_config_far_away('configs/poly.py')
    vis = Toto(cfg)

    s = torch.load('s.pth')
    vis.set_vis_state_dict(s)
