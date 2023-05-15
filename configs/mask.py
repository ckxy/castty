classes = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


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


test_data = dict(
    data_loader=dict(
        batch_size=1,
        serial_batches=True,
        num_threads=0,
        pin_memory=False,
        collator=[
            dict(type='ListCollateFN', names=('image_meta', 'mask_meta')),
        ]
    ),
    dataset=dict(
        reader=dict(type='VOCSegReader', use_pil=False, root='../datasets/voc/VOCdevkit/VOC2012', split='val', classes=classes),
        # reader=dict(type='SBDReader', use_pil=False, root='../datasets/voc/benchmark_RELEASE/dataset', split='train', classes=classes),
        # reader=dict(type='MHPV1Reader', use_pil=False, root='../datasets/LV-MHP-v1', split='train'),
        # reader=dict(type='LabelmeMaskReader', use_pil=False, root='../datasets/waterleveld', classes=['__background__', 'wlr']),
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
            # dict(type='Mosaic', internodes=[
            #     dict(type='MixUp', internodes=[
            #         dict(type='DataSource'),
            #     ]),
            # ]),
            # dict(type='ChooseABranchByID', branchs=[
            #     dict(type='MixUp', internodes=[
            #         dict(type='DataSource'),
            #     ]),
            #     dict(type='DataSource'),
            # ]),
            # dict(type='CopyTag', src_tag='image', dst_tag='ori_image'),
            # dict(type='ToCV2Image'),
            # dict(type='Padding', padding=(20, 30, 40, 50), fill=50, padding_mode='reflect'),
            # dict(type='PaddingBySize', size=(416, 416), fill=(50, 50, 50), padding_mode='constant', center=True),
            # dict(type='PaddingByStride', stride=100, fill=(50, 50, 50), padding_mode='constant', center=False, one_way='orward'),
            # dict(type='Resize', size=(640, 640), keep_ratio=False, short=False),
            # dict(type='WarpResize', size=(416, 416), expand=True, keep_ratio=True, short=False, one_way='forward'),
            # dict(type='ResizeAndPadding', 
            #     resize=dict(
            #         type='WarpResize',
            #         size=(416, 416),
            #         keep_ratio=True,
            #         short=False,
            #         # one_way='forward'
            #     ),
            #     padding=dict(
            #         # type='PaddingBySize',
            #         # size=(416, 416),
            #         type='PaddingByStride',
            #         stride=100,
            #         fill=(0, 0, 0), 
            #         padding_mode='constant',
            #         center=False
            #     ),
            #     # one_way='forward'
            # ),
            # dict(type='Warp', ccs=True, internodes=[
            #     dict(type='WarpPerspective', expand=True, ccs=True),
            #     # dict(type='WarpStretch', rw=(0.5, 1.5), rh=(0.5, 1.5)),
            #     # dict(type='WarpScale', r=(0.5, 1.4)),
            #     # dict(type='WarpRotate', angle=(-30, 30)),
            #     # dict(type='WarpShear', ax=(-30, 30), ay=(-30, 30)),
            #     # dict(type='WarpTranslate', rw=(-0.2, 0.2), rh=(-0.2, 0.2)),
            #     # dict(type='WarpResize', size=(416, 416), expand=False, keep_ratio=True),
            # ]),
            # dict(type='WarpTranslate', rw=(-0.2, 0.2), rh=(-0.2, 0.2), expand=True),
            # dict(type='ChooseOne', branchs=[
            #     dict(type='Bamboo', internodes=[
            #         dict(type='WarpRotate', angle=(0, 30)),
            #     ]),
            #     dict(type='WarpRotate', angle=(-30, 0)),
            # ]),
            # dict(type='ChooseSome', num=4,branchs=[
            #     dict(type='Bamboo', internodes=[
            #         dict(type='WarpStretch', rw=(0.5, 1.5), rh=(1, 1))
            #     ]),
            #     dict(type='WarpStretch', rw=(1, 1), rh=(0.5, 1.5)),
            #     dict(type='Bamboo', internodes=[
            #         dict(type='WarpRotate', angle=(0, 30)),
            #     ]),
            #     dict(type='WarpRotate', angle=(-30, 0)),
            # ]),
            # dict(type='Rot90', k=[1, 2, 3]),
            # dict(type='ToGrayscale'),
            # dict(type='BrightnessEnhancement', brightness=(0.5, 0.5)),
            # dict(type='ContrastEnhancement', contrast=(0.5, 0.5)),
            # dict(type='SaturationEnhancement', saturation=(0.5, 0.5)),
            # dict(type='HueEnhancement', hue=(0.5, 0.5)),
            # dict(type='Flip', horizontal=True),
            # dict(type='Crop', size=(200, 200)),
            # dict(type='CenterCrop', size=(200, 200)),
            # dict(type='RandomErasing', offset=False, value=(0, 0, 0)),
            # dict(type='GridMask', offset=True),
            # dict(type='AdaptiveCrop'),
            # dict(type='AdaptiveTranslate'),
            # dict(type='RandomAreaCrop'),
            # dict(type='Padding', padding=(100, 200, 300, 400)),
            # dict(type='MinIOGCrop', threshs=(-1, 0.1, 0.3, 0.5, 0.7, 0.9)),
            # dict(type='GridMask', use_w=True, use_h=True, rotate=0, offset=False, invert=False, ratio=0.5),
            dict(type='ToPILImage'),
            dict(type='ToTensor'),
            # dict(type='EraseContour', one_way='forward'),
        ],
    ),
)
