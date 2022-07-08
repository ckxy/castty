import os
import cv2
import math
import time
import torch
import ntpath
import random
from PIL import Image
import numpy as np
from tqdm import tqdm

from part_of_hitogata.configs import load_config, load_config_far_away
from part_of_hitogata.visualization.toto import Toto


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    cfg = load_config_far_away('configs/poly.py')
    vis = Toto(cfg)

    # Progress
    pbar_epoch = tqdm(range(5))
    for i in pbar_epoch:
        pbar_step = tqdm(range(10), leave=False)
        for j in pbar_step:
            time.sleep(0.1)
            progress_data = dict(
                progress=dict(
                    epoch=(pbar_epoch.format_dict, ''),
                    step=(pbar_step.format_dict, 'train'),
                ),
            )
            vis.visualize(progress_data, i, 10 * i + j)

    # Line
    print('Line')
    for i in range(5):
        for j in range(10):
            if (10 * i + j) % 2 == 0:
                vis_dict = dict(
                    line=dict(
                        l1=dict(
                            y=2 * i,
                            save=True,
                            opts=dict(title='l1'),
                        ),
                    ),
                )
                vis.visualize(vis_dict, i, 10 * i + j)

                vis_dict = dict(
                    line=dict(
                        l3=dict(
                            y=3 * i,
                            use_step=False,
                            save=True,
                            opts=dict(title='l3'),
                        ),
                    ),
                )
                vis.visualize(vis_dict, i, 10 * i + j)


    vis_dict = dict(
        line=dict(
            l2=dict(
                x=np.arange(10),
                y=np.random.randn(10),
                save=True,
                opts=dict(title='l2'),
            ),
        ),
    )
    vis.visualize(vis_dict, 0, 0)

    # Image
    imgs = [Image.new('RGB', (256, 256), color=(255 - i * 20, 0, i * 20)) for i in range(6)]

    vis_dict = dict(
        image=dict(
            img=dict(
                images=imgs,
                nrow=3,
                save=True,
                opts=dict(title='img'),
            ),
        ),
    )
    vis.visualize(vis_dict, 0, 0)

    # Text
    print('Text')

    text = 'a\nbb\nccc'

    vis_dict = dict(
        text=dict(
            ap=dict(
                text=text,
                save=True,
                opts=dict(title='ap'),
            ),
        ),
    )
    vis.visualize(vis_dict, 0, 0)

    # Heatmap
    print('Heatmap')

    vis_dict = dict(
        heatmap=dict(
            heat=dict(
                x=np.random.rand(256, 256),
                save=True,
                opts=dict(title='heat'),
            ),
        ),
    )
    vis.visualize(vis_dict, 0, 0)

    # Histogram
    print('Histogram')

    vis_dict = dict(
        histogram=dict(
            hist=dict(
                x=np.random.rand(1000),
                save=True,
                opts=dict(title='hist', numbins=20),
            ),
        ),
    )
    vis.visualize(vis_dict, 0, 0)

    # Bar
    print('Bar')

    vis_dict = dict(
        bar=dict(
            counts=dict(
                x=np.random.rand(3),
                save=True,
                opts=dict(
                    title='counts',
                    legend=['The Netherlands', 'France', 'United States'],
                ),
            ),
        ),
    )
    vis.visualize(vis_dict, 0, 0)

    # Scatter
    print('Scatter')

    vis_dict = dict(
        scatter=dict(
            s=dict(
                x=np.random.rand(10, 2),
                y=np.ones(10),
                save=True,
                opts=dict(title='s'),
            ),
        ),
    )
    vis.visualize(vis_dict, 0, 0)

    # Boxplot
    print('Boxplot')

    for i in range(5):
        for j in range(10):
            if (10 * i + j) % 2 == 0:
                vis_dict = dict(
                    boxplot=dict(
                        b1=dict(
                            x=random.random(),
                            update=True,
                            save=True,
                            opts=dict(title='b1'),
                        ),
                    ),
                )
                vis.visualize(vis_dict, i, 10 * i + j)
                # time.sleep(0.2)

    vis_dict = dict(
        boxplot=dict(
            b2=dict(
                x=np.random.rand(100),
                save=True,
                opts=dict(title='b2'),
            ),
        ),
    )
    vis.visualize(vis_dict, 0, 0)

    # Wafer
    print('Wafer')
    vis_dict = dict(
        wafer=dict(
            w=dict(
                x=np.asarray([19, 26, 55]),
                save=True,
                opts=dict(title='w', legend=['Residential', 'Non-Residential', 'Utility']),
            ),
        ),
    )
    vis.visualize(vis_dict, 0, 0)

    # StackedLines
    print('Stackedlines')

    for i in range(5):
        for j in range(10):
            if (10 * i + j) % 2 == 0:
                vis_dict = dict(
                    stackedlines=dict(
                        sl1=dict(
                            y=np.array([i, 2 * i, 3 * i]),
                            save=True,
                            opts=dict(title='sl1'),
                        ),
                    ),
                )
                vis.visualize(vis_dict, i, 10 * i + j)


    vis_dict = dict(
        stackedlines=dict(
            sl2=dict(
                x=np.repeat(np.arange(10)[..., np.newaxis], 3, axis=-1),
                y=np.abs(np.random.randn(10, 3)),
                save=True,
                opts=dict(title='sl2'),
            ),
        ),
    )
    vis.visualize(vis_dict, 0, 0)

    print('saving')
    s = vis.get_vis_state_dict()
    torch.save(s, 's.pth')
