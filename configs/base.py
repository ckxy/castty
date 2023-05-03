general = dict(
    experiment_name='test',
    checkpoints_dir='./',
)

visualizer = dict(
    visdom=dict(
        server='http://localhost',
        port=8097,
        username='',
        password=''
    ),
)

bbox = [306, 308, 696, 870]

label = [[1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [1, 0]]

point = [[307.17, 508.886 ],
 [584.075, 506.654 ],
 [365.906, 687.265 ],
 [451.5053, 684.0441],
 [431.5594, 870.5972]]

poly = [[321.906, 570.918 ],
 [323.7967, 556.6462],
 [334.5908, 548.2142],
 [360.3877, 551.5268],
 [379.9596, 568.8308],
 [358.4041, 575.5783],
 [335.571, 578.0543],
 [328.2922, 575.4066]]

mask = [[459.847, 566.848 ],
 [472.9587, 552.1189],
 [491.2339, 545.2528],
 [516.5079, 549.7994],
 [540.8568, 558.6542],
 [518.6608, 569.9822],
 [494.3823, 574.3677],
 [477.0045, 571.1513]]


mode = [
    'label',
    'bbox',
    'mask',
    'point',
    'poly'
]

test_data = dict(
    data_loader=dict(
        batch_size=1,
        serial_batches=True,
        num_threads=0,
        pin_memory=False,
        collator=[
            dict(type='ListCollateFN', names=('image_meta',)),
            dict(type='LabelCollateFN', names=('label',)),
            dict(type='BboxCollateFN', names=('bbox',)),
            dict(type='ListCollateFN', names=('bbox_meta',)),
            dict(type='ListCollateFN', names=('mask_meta',)),
            dict(type='ListCollateFN', names=('point_meta',)),
            dict(type='ListCollateFN', names=('poly', 'poly_meta')),
        ]
    ),
    dataset=dict(
        reader=dict(type='FondReader', mode=mode, image='images/test.jpg', label=label, bbox=bbox, mask=mask, point=point, poly=poly, use_pil=True),
        internodes=[
            dict(type='DataSource'),
            # dict(type='MixUp', internodes=[
            #     dict(type='DataSource'),
            # ]),
            # dict(type='Mosaic', internodes=[
            #     dict(type='DataSource'),
            # ]),
            # dict(type='CutMix', internodes=[
            #     dict(type='DataSource'),
            # ]),
            # dict(type='ChooseOne', branchs=[
            #     dict(type='BrightnessEnhancement', brightness=(0.5, 0.5)),
            #     dict(type='HueEnhancement', hue=(0.5, 0.5)),
            # ]),
            # dict(type='ChooseSome', branchs=[
            #     dict(type='BrightnessEnhancement', brightness=(0.5, 0.5)),
            #     dict(type='ContrastEnhancement', contrast=(0.5, 0.5)),
            #     dict(type='SaturationEnhancement', saturation=(0.5, 0.5)),
            #     dict(type='HueEnhancement', hue=(0.5, 0.5)),
            # ]),
            # dict(type='ToGrayscale', p=0.5),
            # dict(type='ToGrayscale'),
            # dict(type='BrightnessEnhancement', brightness=(0.5, 0.5)),
            # dict(type='ContrastEnhancement', contrast=(0.5, 0.5)),
            # dict(type='SaturationEnhancement', saturation=(0.5, 0.5)),
            # dict(type='HueEnhancement', hue=(0.5, 0.5)),
            # dict(type='ToCV2Image'),
            # dict(type='ToPILImage'),
            # dict(type='Crop', size=(400, 150)),
            # dict(type='AdaptiveCrop', one_way='forward'),
            # dict(type='MinIOUCrop', threshs=(-1, 0.1, 0.3, 0.5, 0.7, 0.9), use_base_filter=False),
            # dict(type='MinIOGCrop', threshs=(-1, 0.1, 0.3, 0.5, 0.7, 0.9)),
            # dict(type='CenterCrop', size=(480, 480), use_base_filter=False),
            # dict(type='RandomAreaCrop'),
            # dict(type='EastRandomCrop', min_crop_side_ratio=0.4),
            # dict(type='WestRandomCrop', min_crop_side_ratio=0.4),
            # dict(type='RandomCenterCropPad', size=(512, 512), ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3)),
            # dict(type='RandomErasing', offset=False, value=(0, 0, 0)),
            # dict(type='GridMask', offset=True),
            # dict(type='Flip', horizontal=True),
            # dict(type='SwapChannels', swap=(2, 1, 0)),
            # dict(type='RandomSwapChannels'),
            # dict(type='Padding', padding=(20, 30, 40, 50), fill=(50, 50, 50), padding_mode='reflect'),
            # dict(type='PaddingBySize', size=(1200, 1200), fill=(0, 0, 0), padding_mode='constant', center=True),
            # dict(type='PaddingByStride', stride=140, fill=(0, 0, 0), padding_mode='constant', center=True),
            # dict(type='RandomExpand', ratio=2),
            # dict(type='ResizeAndPadding', 
            #     resize=dict(
            #         type='WarpResize',
            #         size=(512, 512),
            #         keep_ratio=True,
            #         short=False,
            #     ),
            #     padding=dict(
            #         # type='PaddingBySize',
            #         # size=(600, 600),
            #         type='PaddingByStride',
            #         stride=100,
            #         fill=(0, 0, 0), 
            #         padding_mode='constant',
            #         center=True,
            #         # use_base_filter=False
            #     ),
            #     # one_way='forward'
            # ),
            # dict(type='Rescale', ratio_range=(0.5, 3), mode='range'),
            # dict(type='RescaleLimitedByBound', long_size_bound=1280, short_size_bound=640, ratio_range=(0.5, 3), mode='range'),
            # dict(type='Rot90', k=[1, 2, 3]),
            # dict(type='Warp', expand=True, ccs=True, internodes=[
            #     dict(type='WarpPerspective'),
            #     dict(type='WarpStretch', rw=(1.5, 1.5), rh=(0.5, 0.5)),
            #     dict(type='WarpScale', r=(1.2, 1.2)),
            #     dict(type='WarpShear', ax=(-45, -45), ay=(15, 15)),
            #     dict(type='WarpRotate', angle=(-30, -30)),
            #     # dict(type='WarpTranslate', rw=(-0.2, -0.2), rh=(0.2, 0.2)),
            #     # dict(type='WarpResize', size=(416, 416), keep_ratio=True),
            # ]),
            # dict(type='Warp', expand=True, ccs=True, internodes=[
            #     dict(type='WarpStretch', rw=(1.5, 1.5), rh=(0.5, 0.5), p=0.5),
            #     dict(type='WarpScale', r=(2, 2)),
            # ]),
            # dict(type='WarpPerspective', expand=True, ccs=True),
            # dict(type='WarpScale', r=(0.5, 2), expand=True),
            # dict(type='WarpStretch', rw=(1.5, 1.5), rh=(0.5, 0.5), expand=True),
            # dict(type='WarpRotate', angle=(-30, -30), expand=True),
            # dict(type='WarpShear', ax=(-30, -30), ay=(15, 15), expand=True),
            # dict(type='WarpTranslate', rw=(-0.2, -0.2), rh=(0.2, 0.2)),
            # dict(type='WarpResize', size=(416, 416), expand=False, keep_ratio=True),
            # dict(type='AdaptiveTranslate'),
            # dict(type='TPSStretch', segment=2),
            # dict(type='TPSDistort', segment=4, resize=True),
            dict(type='ToTensor'),
            # dict(type='Normalize', mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            # dict(type='To1CHTensor'),
        ],
    ),
)

