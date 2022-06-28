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

priors = dict(
    image_size=300,
    shapes=(38, 19, 10, 5, 3, 1),
    steps=(8, 16, 32, 64, 100, 300),
    min_sizes=(30, 60, 111, 162, 213, 264),
    max_sizes=(60, 111, 162, 213, 264, 315),
    aspect_ratios=((2, 3), (2, 3), (2, 3), (2, 3), 2, 2),
    variances=(0.1, 0.2),
    clip=True,
)

test_data = dict(
    data_loader=dict(
        batch_size=1,
        serial_batches=True,
        num_threads=0,
        pin_memory=False,
        collator=[
            ('BboxCollateFN', dict(names=('bbox',))),
            # ('ListCollateFN', dict(names=('neg_category_ids', 'not_exhaustive_category_ids'))),
            ('ListCollateFN', dict(names=('ga_bbox', 'ga_index'))),
            # ('NanoCollateFN', dict()),
            # ('SYoloCollateFN', dict())
        ]
    ),
    dataset=dict(
        # reader=('COCOAPIReader', dict(set_path='../datasets/coco/annotations/instances_train2017.json', img_root='../datasets/coco/train2017')),
        # reader=('VOCLikeSegReader', dict(root='/home/ubuntu/datasets/water_seg/water_v2', cls_and_clr=(('__background__', 0), ('water', 255)), split='val')),
        # reader=('COCOAPIReader', dict(set_path='/home/ubuntu/datasets/coco/annotations/instances_val2017.json', img_root='/home/ubuntu/datasets/coco/images/val2017')),
        # reader=('VOCReader', dict(root='../datasets/voc//VOCdevkit/VOC2012', split='trainval', filter_difficult=False, classes=classes)),
        # reader=('LVISAPIReader', dict(set_path='/home/ubuntu/datasets/lvis/lvis_v1_train.json', img_root='/home/ubuntu/datasets/lvis/train2017')),
        # reader=('Market1501AttritubesReader', dict(root='/home/ubuntu/datasets/Market-1501', group='train', mode='ab')),
        internodes=[
            ('Register', dict()),
            # ('EraseTags', dict(tags='bbox_ignore')),
            # ('EraseTags', dict(tags=['neg_category_ids', 'not_exhaustive_category_ids'])),
            # ('VOCReader', dict(root='/home/ubuntu/VOCdevkit/VOC2007', split='trainval', filter_difficult=False, classes=classes)),
            # ('EraseTags', dict(tags='difficult')),
            # ('SBDReader', dict(root='/home/ubuntu/VOCdevkit/benchmark_RELEASE/dataset', split='train', classes=classes)),
            # ('VOCSegReader', dict(root='/home/ubuntu/VOCdevkit/VOC2012', split='seg11valid', classes=classes)),
            # ('RandomErasing', dict(value=(50, 50, 50))),
            # ('GaussianBlur', dict(radius=2)),
            # ('GridMask', dict(use_w=True, use_h=True, rotate=30, offset=True, invert=False, ratio=0.5)),
            # ('RandomCrop', dict(size=(300, 200))),
            # ('AdaptiveRandomCrop', dict()),
            # ('AdaptiveRandomTranslate', dict()),
            # ('MinIOGCrop', dict(threshs=(-1, 0.1, 0.3, 0.5, 0.7, 0.9))),
            # ('CenterCrop', dict(size=(300, 200))),
            # ('RandomAreaCrop', dict()),
            # ('RandomFlip', dict(horizontal=True, p=1)),
            # ('ToCV2Image', dict()),
            ('Warp', dict(
                internodes=[
                    # ('WarpScale', dict(r=(0.5, 1.4), p=0.5)),
                    # ('WarpTranslate', dict(rw=(-0.2, 0.2), rh=(-0.2, 0.2), p=0.5)),
                    ('WarpResize', dict(size=(544, 544), keep_ratio=True)),
                ]
            )),
            # ('WarpPerspective', dict(expand=True)),
            # ('WarpResize', dict(size=(416, 416), keep_ratio=True, expand=True)),
            # ('WarpRotate', dict(angle=(-30, 30))),
            # ('WarpScale', dict(r=(0.5, 1.4))),
            # ('WarpStretch', dict(rw=(0.5, 1.5), rh=(0.5, 1.5))),
            # ('WarpTranslate', dict(rw=(-0.2, 0.2), rh=(-0.2, 0.2))),
            # ('WarpShear', dict(ax=(-30, 30), ay=(-30, 30))),
            # ('ToPILImage', dict()),
            # ('BrightnessEnhancement', dict(brightness=(0.8, 1.2), p=0.5)),
            # ('ContrastEnhancement', dict(contrast=(0.6, 1.4), p=0.5)),
            # ('SaturationEnhancement', dict(saturation=(0.5, 1.2), p=0.5)),
            # ('ResizeAndPadding', dict(size=(416, 416), keep_ratio=True, warp=True)),
            # ('Padding', dict(padding=(20, 30, 40, 50), fill=50, padding_mode='edge')),
            # ('PaddingByStride', dict(stride=32)),
            # ('CropROKP', dict(mode='box', expand=(0.8, 1.5))),
            # ('FilterBboxByLength', dict(min_w=40, min_h=40)),
            # ('FilterBboxByLengthRatio', dict(min_w=0.1, min_h=0.1)),
            # ('FilterBboxByArea', dict(min_a=400)),
            # ('FilterBboxByAreaRatio', dict(min_a=0.05)),
            # ('FilterBboxByAspectRatio', dict(aspect_ratio=(0.8, 1.2))),
            ('ToTensor', dict()),
            # ('SwapChannels', dict(swap=(2, 1, 0))),
            # ('RandomSwapChannels', dict()),
            # ('Normalize', dict(mean=(103.53 / 255, 116.28 / 255, 123.675 / 255), std=(57.375 / 255, 57.12 / 255, 58.395 / 255))),
            ('CalcNanoGrids', dict(scale=5, top_k=9, strides=(8, 16, 32), num_classes=len(classes), analysis=True)),
            # ('CalcGrids', dict(strides=(8, 16, 32), gt_per_grid=2, num_classes=len(classes), sml_thresh=(32, 96), analysis=True, centerness=0)),
            # ('CalcFCOSGrids', dict(strides=(8, 16, 32), multiples=(4, 8), analysis=True)),
            # ('CalcSSDGrids', dict(num_classes=len(classes), priors=priors)),
        ],
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
        ]
    ),
    dataset=dict(
        reader=dict(type='LVISAPIReader', set_path='../datasets/coco/annotations/lvis_v1_val.json', img_root='../datasets/coco'),
        # reader=dict(type='COCOAPIReader', set_path='../datasets/coco/annotations/instances_val2017.json', img_root='../datasets/coco/val2017'),
        # reader=dict(type='VOCReader', use_pil=True, root='../datasets/voc/VOCdevkit/VOC2007', split='trainval', filter_difficult=False, classes=classes),
        # reader=dict(
        #     type='CatReader', 
        #     internodes=(
        #         dict(type='VOCReader', root='../datasets/voc/VOCdevkit/VOC2007', split='trainval', filter_difficult=False, classes=classes),
        #         dict(type='VOCReader', root='../datasets/voc/VOCdevkit/VOC2012', split='trainval', filter_difficult=False, classes=classes),
        #     ),
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
            # dict(type='ChooseABranchByID', branchs=[
            #     dict(type='MixUp', internodes=[
            #         dict(type='DataSource'),
            #     ]),
            #     dict(type='DataSource'),
            # ]),
            # dict(type='CopyTag', src_tag='image', dst_tag='ori_image'),
            # dict(type='ToCV2Image'),
            # dict(type='Padding', padding=(20, 30, 40, 50), fill=50, padding_mode='reflect'),
            # dict(type='PaddingBySize', size=(416, 416), fill=50, padding_mode='constant', center=True),
            # dict(type='PaddingByStride', stride=32, fill=(50, 50, 50), padding_mode='constant', center=False, one_way='orward'),
            # dict(type='Resize', size=(416, 416), keep_ratio=True, short=False),
            # dict(type='WarpResize', size=(416, 416), expand=True, keep_ratio=True, short=False, one_way='forward'),
            # dict(type='ResizeAndPadding', 
            #     resize=dict(
            #         type='WarpResize',
            #         size=(416, 416),
            #         keep_ratio=True,
            #         short=False,
            #     ),
            #     padding=dict(
            #         type='PaddingBySize',
            #         size=(416, 416),
            #         # type='PaddingByStride',
            #         # stride=32,
            #         fill=(0, 0, 0), 
            #         padding_mode='constant',
            #         center=False
            #     ),
            #     one_way='forward'
            # ),
            # dict(type='Warp', p=0.5, ccs=True, internodes=[
            #     # dict(type='WarpPerspective', expand=True, ccs=True),
            #     # dict(type='WarpStretch', rw=(0.5, 1.5), rh=(0.5, 1.5)),
            #     # dict(type='WarpScale', r=(0.5, 1.4)),
            #     # dict(type='WarpRotate', angle=(-30, 30)),
            #     # dict(type='WarpShear', ax=(-30, 30), ay=(-30, 30)),
            #     dict(type='WarpTranslate', rw=(-0.2, 0.2), rh=(-0.2, 0.2)),
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
            # dict(type='ToGrayscale'),
            # dict(type='BrightnessEnhancement', brightness=(0.5, 0.5)),
            # dict(type='ContrastEnhancement', contrast=(0.5, 0.5)),
            # dict(type='SaturationEnhancement', saturation=(0.5, 0.5)),
            # dict(type='HueEnhancement', hue=(0.5, 0.5)),
            # dict(type='Flip', horizontal=True),
            # dict(type='AdaptiveCrop'),
            # dict(type='AdaptiveTranslate'),
            # dict(type='MinIOGCrop', threshs=(-1, 0.1, 0.3, 0.5, 0.7, 0.9)),
            # dict(type='GridMask', use_w=True, use_h=True, rotate=0, offset=False, invert=False, ratio=0.5),
            # dict(type='ToPILImage'),
            dict(type='ToTensor'),
        ],
    ),
)
