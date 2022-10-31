classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

# classes = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
#            'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


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
            dict(type='BboxCollateFN', names=('bbox',)),
            dict(type='ListCollateFN', names=('bbox_meta',)),
            # dict(type='ListCollateFN', names=('bbox_meta', 'point', 'point_meta')),
            # dict(type='ListCollateFN', names=('bbox_meta', 'ga_bbox', 'ga_index')),
            # dict(type='NanoCollateFN')
        ]
    ),
    dataset=dict(
        # reader=dict(type='LVISAPIReader', set_path='../datasets/coco/annotations/lvis_v1_val.json', img_root='../datasets/coco'),
        # reader=dict(type='COCOAPIReader', set_path='../datasets/coco/annotations/instances_val2017.json', img_root='../datasets/coco/val2017'),
        # reader=dict(type='COCOAPIReader', use_keypoint=True, set_path='../datasets/coco/annotations/person_keypoints_val2017.json', img_root='../datasets/coco/val2017'),
        reader=dict(type='VOCReader', use_pil=True, root='../datasets/voc/VOCdevkit/VOC2007', split='trainval', filter_difficult=False, classes=classes),
        # reader=dict(
        #     type='CatReader', 
        #     internodes=(
        #         dict(type='VOCReader', root='../datasets/voc/VOCdevkit/VOC2007', split='trainval', filter_difficult=False, classes=classes),
        #         dict(type='VOCReader', root='../datasets/voc/VOCdevkit/VOC2012', split='trainval', filter_difficult=False, classes=classes),
        #     ),
        #     output_gid=True,
        # ),
        internodes=[
            dict(type='DataSource'),
            # dict(type='MixUp', internodes=[
            #     dict(type='DataSource'),
            # ]),
            # dict(type='Mosaic', internodes=[
            #     dict(type='DataSource'),
            # ]),
            # dict(type='Mosaic', internodes=[
            #     dict(type='MixUp', internodes=[
            #         dict(type='DataSource'),
            #     ]),
            # ]),
            # dict(
            #     type='ChooseABranchByID', 
            #     branchs=[
            #         dict(
            #             type='MixUp', 
            #             internodes=[dict(type='DataSource')]
            #         ),
            #         dict(type='DataSource'),
            #     ],
            #     tag='branch_id',
            # ),
            # dict(
            #     type='ChooseABranchByID', 
            #     branchs=[
            #         dict(type='WarpRotate', angle=(-30, -30), expand=True),
            #         dict(type='WarpRotate', angle=(30, 30), expand=True),
            #     ],
            #     tag='group_id',
            # ),
            # dict(type='CopyTag', src_tag='image', dst_tag='ori_image'),
            # dict(type='ToCV2Image'),
            # dict(type='AdaptiveCrop'),
            # dict(type='Resize', size=(416, 416), keep_ratio=True, short=False),
            # dict(type='Rescale', ratio_range=(0.5, 2)),
            # dict(type='WarpResize', size=(416, 416), expand=True, keep_ratio=True, short=False, one_way='forward'),
            # dict(type='Padding', padding=(20, 30, 40, 50), fill=(50, 50, 50), padding_mode='reflect'),
            # dict(type='PaddingBySize', size=(416, 416), fill=(50, 50, 50), padding_mode='constant', center=False),
            # dict(type='PaddingByStride', stride=100, fill=(50, 50, 50), padding_mode='constant', center=False),
            # dict(type='ResizeAndPadding', 
            #     resize=dict(
            #         type='WarpResize',
            #         size=(512, 512),
            #         keep_ratio=True,
            #         short=False,
            #     ),
            #     padding=dict(
            #         type='PaddingBySize',
            #         size=(600, 600),
            #         # type='PaddingByStride',
            #         # stride=100,
            #         fill=(0, 0, 0), 
            #         padding_mode='constant',
            #         center=True
            #     ),
            #     one_way='forward'
            # ),
            # dict(type='Warp', expand=True, ccs=True, internodes=[
            #     dict(type='WarpPerspective'),
            #     dict(type='WarpStretch', rw=(1.5, 1.5), rh=(0.5, 0.5)),
            #     dict(type='WarpScale', r=(2, 2)),
            #     dict(type='WarpShear', ax=(-45, -45), ay=(15, 15), p=0.5),
            #     dict(type='WarpRotate', angle=(-30, -30)),
            #     # dict(type='WarpTranslate', rw=(-0.2, -0.2), rh=(0.2, 0.2)),
            #     # dict(type='WarpResize', size=(416, 416), keep_ratio=True),
            # ]),
            # dict(type='WarpPerspective', expand=True, ccs=True),
            # dict(type='WarpScale', r=(0.5, 2), expand=True),
            # dict(type='WarpStretch', rw=(1.5, 1.5), rh=(0.5, 0.5), expand=True),
            # dict(type='WarpRotate', angle=(-30, -30), expand=True),
            # dict(type='WarpShear', ax=(-30, -30), ay=(15, 15), expand=True),
            # dict(type='WarpTranslate', rw=(-0.2, -0.2), rh=(0.2, 0.2), expand=True),
            # dict(type='WarpResize', size=(416, 416), expand=False, keep_ratio=True),
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
            # dict(type='ToGrayscale'),
            # dict(type='BrightnessEnhancement', brightness=(0.5, 0.5)),
            # dict(type='ContrastEnhancement', contrast=(0.5, 0.5)),
            # dict(type='SaturationEnhancement', saturation=(0.5, 0.5)),
            # dict(type='HueEnhancement', hue=(0.5, 0.5)),
            # dict(type='Flip', horizontal=False),
            # dict(type='AdaptiveCrop'),
            # dict(type='AdaptiveTranslate'),
            # dict(type='MinIOGCrop', threshs=(-1, 0.1, 0.3, 0.5, 0.7, 0.9)),
            # dict(type='FilterBboxByLength', min_w=50, min_h=50),
            # dict(type='FilterBboxByArea', min_a=7000),
            # dict(type='FilterBboxByLengthRatio', min_w=0.1, min_h=0.1),
            # dict(type='FilterBboxByAreaRatio', min_a=0.05),
            # dict(type='FilterBboxByAspectRatio', aspect_ratio=(0.5, 2)),
            # dict(type='GridMask', use_w=True, use_h=True, rotate=0, offset=False, invert=False, ratio=0.5),
            # dict(type='SwapChannels', swap=(2, 1, 0)),
            # dict(type='RandomSwapChannels'),
            dict(type='WestRandomCrop', min_crop_side_ratio=0.1),
            # dict(type='ToPILImage'),
            dict(type='ToTensor'),
            # dict(type='CalcCenterNetGrids', ratio=0.25, num_classes=1),
            # dict(type='CalcNanoGrids', scale=5, top_k=9, strides=(8, 16, 32), num_classes=len(classes), analysis=True),
        ],
    ),
)
