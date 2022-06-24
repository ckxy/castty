import os
import cv2
import math
import time
import torch
import ntpath
import random
from PIL import Image
import numpy as np
from configs import load_config, load_config_far_away
from tqdm import tqdm
from visualization.visualizer import Visualizer
# from visualization.visualizer_bk import Visualizer


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    cfg = load_config('spnx')
    vis = Visualizer(cfg)

    s = torch.load('s.pth')
    vis.set_vis_state_dict(s)
