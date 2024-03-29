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
        max_size=-1,
        # reader=('COCOAPIReader', dict(set_path='../datasets/coco/annotations/instances_train2017.json', img_root='../datasets/coco/train2017')),
        # reader=('VOCLikeSegReader', dict(root='/home/ubuntu/datasets/water_seg/water_v2', cls_and_clr=(('__background__', 0), ('water', 255)), split='val')),
        # reader=('COCOAPIReader', dict(set_path='/home/ubuntu/datasets/coco/annotations/instances_val2017.json', img_root='/home/ubuntu/datasets/coco/images/val2017')),
        reader=('VOCReader', dict(root='../datasets/voc//VOCdevkit/VOC2012', split='trainval', filter_difficult=False, classes=classes)),
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

# test_data = dict(
#     data_loader=dict(
#         batch_size=1,
#         serial_batches=True,
#         num_threads=0,
#         pin_memory=False
#     ),
#     dataset=dict(
#         max_size=-1,
#         internodes=[
#             ('WFLWReader', dict(root='/home/ubuntu/datasets/WFLW/WFLW', txt_path='WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt')),
#             ('EraseTags', dict(tags='bbox')),
#             # ('WFLWPPReader', dict(txt_path='/home/ubuntu/datasets/WFLW/test_data/list.txt')),
#             # ('RandomFlip', dict(horizontal=True, p=1, mapping='/home/ubuntu/datasets/WFLW/Mirror98.txt')),
#             # ('AttributeSelector', dict(indices=[1, 2, 3, 4, 5])),
#             ('CropROKP', dict(mode='point', expand=(1, 1.25))),
#             # ('CropROKP', dict(mode='img', expand=(1, 1))),
#             ('ResizeAndPadding', dict(size=(512, 512), keep_ratio=False, warp=False)),
#             ('NormCoor', dict(size=(512, 512))),
#             # ('NormCoor', dict(no_forward=True)),
#             ('ToTensor', dict()),
#             # ('CalcEulerAngles', dict(tp=(33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16), path='/home/ubuntu/datasets/WFLW/landmarks_3D.txt'))
#         ],
#     ),
# )

# test_data = dict(
#     data_loader=dict(
#         batch_size=1,
#         serial_batches=True,
#         num_threads=0,
#         pin_memory=False,
#     ),
#     dataset=dict(
#         max_size=-1,
#         internodes=[
#             ('MPIIH5Reader', dict(root='/home/ubuntu/datasets/mpii', set_path='/home/ubuntu/datasets/mpii/annot_h5/train.h5')),
#             # ('LSPReader', dict(root='/home/ubuntu/datasets/lsp', set_path='/home/ubuntu/datasets/lsp/LEEDS_annotations.json', is_test=False)),
#             ('RandomRescale', dict()),
#             ('CropROKP', dict(mode='box', expand=(1, 1))),
#             ('EraseTags', dict(tags=('mpii_scale', 'mpii_length'))),
#             ('ResizeAndPadding', dict(size=(256, 256), keep_ratio=True, warp=False)),
#             ('ToTensor', dict()),
#             ('CalcHeatmapByPoint', dict(sigma=2, ratio=0.25, resample=False)),
#         ],
#     ),
# )


# test_data = dict(
#     data_loader=dict(
#         batch_size=1,
#         serial_batches=True,
#         num_threads=0,
#         pin_memory=False,
#     ),
#     dataset=dict(
#         name='general',
#         max_size=-1,
#         internodes=[
#             ('Market1501AttritubesReader', dict(root='/home/ubuntu/datasets/Market-1501', group='train', mode='ab')),
#             # ('DukeMTMCAttritubesReader', dict(root='/home/ubuntu/datasets/DukeMTMC-reID', group='train', mode='c')),
#             # ('ImageFolderReader', dict(root='/home/ubuntu/datasets/catdog/val')),
#             # ('Resize', dict(size=(256, 256), keep_ratio=True, short=False)),
#             # ('CenterCrop', dict(size=(224, 224))),
#             ('ToTensor', dict()),
#         ],
#     ),
# )

# test_data = dict(
#     data_loader=dict(
#         batch_size=1,
#         serial_batches=True,
#         num_threads=0,
#         pin_memory=False,
#         # collate_function=('CRAFTCollateFN', dict())
#         collator=[
#             # ('ListCollateFN', dict(names=('quad_affinity', 'quad_char', 'seq'))),
#             ('ListCollateFN', dict(names=('seq', 'quad'))),
#         ]
#     ),
#     dataset=dict(
#         name='general',
#         max_size=-1,
#         internodes=[
#             ('CharSegMLReader', dict(root='/home/ubuntu/Videos/4/craft/IMG_6787')),
#             ('ResizeAndPadding', dict(size=(768, 320), keep_ratio=True, warp=False)),
#             ('ToTensor', dict()),
#             ('Normalize', dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))),
#             ('CalcHeatmapByQuad', dict(ratio=2, thresh=0.35)),
#             ('CalcAffinityQuad', dict()),
#             ('RenameTag', dict(old_name='quad', new_name='quad_char')),
#             ('RenameTag', dict(old_name='quad_affinity', new_name='quad')),
#             ('RenameTag', dict(old_name='heatmap', new_name='heatmap_char')),
#             ('CalcHeatmapByQuad', dict(ratio=2, thresh=0.15)),
#             ('RenameTag', dict(old_name='heatmap', new_name='heatmap_affinity')),
#             ('RenameTag', dict(old_name='quad', new_name='quad_affinity')),
#         ],
#     ),
# )

# test_data = dict(
#     data_loader=dict(
#         batch_size=1,
#         serial_batches=True,
#         num_threads=0,
#         pin_memory=False,
#         collator=[
#             ('BboxCollateFN', dict(names=('bbox',))),
#             ('ListCollateFN', dict(names=('ga_bbox', 'ga_index'))),
#             # ('NanoCollateFN', dict()),
#         ]
#     ),
#     dataset=dict(
#         max_size=-1,
#         # reader=('COCOBboxTxtReader', dict(txt_root='/home/ubuntu/datasets/coco/labels/val2017', img_root='/home/ubuntu/datasets/coco/images/val2017', classes=classes)),
#         reader=('COCOAPIReader', dict(set_path='../datasets/coco/annotations/instances_val2017.json', img_root='../datasets/coco/val2017')),
#         # reader=('VOCReader', dict(root='/home/ubuntu/VOCdevkit/VOC2007', split='trainval', filter_difficult=False, classes=classes)),
#         # reader=('UniReader', dict(internodes=(
#         #     ('LabelmeMaskReader', dict(root='/home/ubuntu/datasets/waterlevel', classes=('__background__', 'wlr'))), 
#         #     ('LabelmeMaskReader', dict(root='/home/ubuntu/datasets/waterlevel/extra', classes=('__background__', 'wlr'))),
#         #     ))
#         # ),
#         internodes=[
#             # ('Register', dict()),
#             # ('EraseTags', dict(tags='bbox_ignore')),
#             # ('ChooseOne', dict(
#             #     internodes=[
#             #         ('MixUp', dict(
#             #             internodes=[
#             #                 ('Register', dict()),
#             #                 ('EraseTags', dict(tags='bbox_ignore')),
#             #                 ('ResizeAndPadding', dict(size=(416, 416), keep_ratio=True, warp=False)),
#             #             ]
#             #         )),
#             #         ('Mosaic', dict(
#             #             internodes=[
#             #                 ('Register', dict()),
#             #             ]
#             #         )),
#             #     ]
#             # )),
#             ('Mosaic', dict(
#                 internodes=[
#                     ('MixUp', dict(
#                         internodes=[
#                             ('Register', dict()),
#                             ('EraseTags', dict(tags='bbox_ignore')),
#                             ('ResizeAndPadding', dict(size=(512, 512), keep_ratio=True, warp=False)),
#                         ]
#                     )),
#                 ]
#             )),
#             # ('MixUp', dict(
#             #     internodes=[
#             #         ('Register', dict()),
#             #         ('ResizeAndPadding', dict(size=(416, 416), keep_ratio=True, warp=False)),
#             #     ]
#             # )),
#             # ('ResizeAndPadding', dict(size=(416, 416), keep_ratio=True, warp=True)),
#             ('ToTensor', dict()),
#             ('CalcNanoGrids', dict(scale=5, top_k=9, strides=(8, 16, 32), num_classes=len(classes), analysis=True)),
#         ],
#     ),
# )

# test_data = dict(
#     data_loader=dict(
#         batch_size=6,
#         serial_batches=False,
#         num_threads=0,
#         pin_memory=False,
#         analyser=('image_analysis', dict(mode='len')),
#         batch_sampler=('StepsSampler', dict(steps=1000))
#     ),
#     dataset=dict(
#         max_size=-1,
#         reader=('C2NReader', dict(root='/home/ubuntu/test/cmte/datasets/half/mix3')),
#         internodes=[
#             ('Register', dict()),
#             ('CMTERandomFlip', dict(p=1)),
#             ('CMTECrop', dict(size=(256, 256))),
#             ('RenameTag', dict(old_name='a_image', new_name='image')),
#             ('ToTensor', dict()),
#             ('Normalize', dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))),
#             ('RenameTag', dict(old_name='image', new_name='a_image')),
#             ('RenameTag', dict(old_name='a_star_image', new_name='image')),
#             ('ToTensor', dict()),
#             ('Normalize', dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))),
#             ('RenameTag', dict(old_name='image', new_name='a_star_image')),
#             ('RenameTag', dict(old_name='b_image', new_name='image')),
#             ('ToTensor', dict()),
#             ('Normalize', dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))),
#             ('RenameTag', dict(old_name='image', new_name='b_image')),
#         ],
#     ),
# )

# test_data = dict(
#     data_loader=dict(
#         batch_size=1,
#         serial_batches=True,
#         num_threads=0,
#         pin_memory=False,
#     ),
#     dataset=dict(
#         max_size=-1,
#         # reader=('MEJSONReader', dict(mode='6s', root='/home/ubuntu/datasets/meter', txt_path='/home/ubuntu/datasets/meter/m2/tr2m2p.txt')),
#         reader=('NewMEJSONReader', dict(root='/home/ubuntu/datasets/meter/res/4', split='train')),
#         internodes=[
#             ('Register', dict()),
#             # ('ToCV2Image', dict()),
#             # ('WarpRotate', dict(angle=(-30, 30), expand=True)),
#             # ('ToPILImage', dict()),
#             # ('CropROKP', dict(mode='box', expand=(1, 1))),
#             # ('CropROKP', dict(mode='point', expand=(2, 4))),
#             ('EraseTags', dict(tags='bbox')),
#             # ('AdaptiveRandomTranslate', dict()),
#             # ('ResizeAndPadding', dict(size=(112, 112), keep_ratio=False, warp=False)),
#             ('ToTensor', dict()),
#             # ('NormCoor', dict(size=(112, 112))),
#             # ('CalcEulerAngles', dict(tp=(0, 1, 2, 3, 4, 5), path='/home/ubuntu/datasets/meter/landmarks_3D6s.txt')),
#             ('CalcHeatmapByPoint', dict(sigma=2, ratio=0.25, resample=False)),
#         ],
#     ),
# )

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
        max_size=-1,
        # reader=dict(type='LVISAPIReader', set_path='../datasets/coco/annotations/lvis_v1_val.json', img_root='../datasets/coco'),
        # reader=dict(type='COCOAPIReader', set_path='../datasets/coco/annotations/instances_val2017.json', img_root='../datasets/coco/val2017'),
        reader=dict(type='VOCReader', use_pil=True, root='../datasets/voc/VOCdevkit/VOC2007', split='trainval', filter_difficult=False, classes=classes),
        # reader=dict(type='CatReader', internodes=(
        #         dict(type='VOCReader', root='../datasets/voc/VOCdevkit/VOC2007', split='trainval', filter_difficult=False, classes=classes),
        #         dict(type='VOCReader', root='../datasets/voc/VOCdevkit/VOC2012', split='trainval', filter_difficult=False, classes=classes),
        #     )
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

# test_data = dict(
#     data_loader=dict(
#         batch_size=1,
#         serial_batches=True,
#         num_threads=0,
#         pin_memory=False,
#         collator=[
#             dict(type='ListCollateFN', names=('poly', 'poly_meta')),
#         ]
#     ),
#     dataset=dict(
#         max_size=-1,
#         reader=dict(type='ICDARDetReader', root='../datasets/ICDAR2015', use_pil=False),
#         internodes=[
#             dict(type='DataSource'),
#             dict(type='ResizeAndPadding', 
#                 resize=dict(
#                     type='Resize',
#                     size=(960, 960),
#                     keep_ratio=True,
#                     short=False,
#                 ),
#                 padding=dict(
#                     type='PaddingBySize',
#                     size=(960, 960),
#                     fill=(0, 0, 0), 
#                     padding_mode='constant',
#                 ),
#                 one_way='forward'
#             ),
#             # dict(type='Flip', horizontal=True, p=0.5),
#             # dict(type='WarpRotate', angle=(-10, 10), expand=True),
#             # dict(type='RescaleLimitedByBound', long_size_bound=1280, short_size_bound=640, ratio_range=(0.5, 3), mode='range'),
#             # dict(type='PSEEncode'),
#             # dict(type='EraseTags', tags=['poly', 'poly_meta']),
#             # dict(type='PSECrop', size=(640, 640), positive_sample_ratio=5.0 / 8.0),
#             dict(type='ToPILImage'),
#             dict(type='ToTensor'),
#         ],
#     ),
# )

# test_data = dict(
#     data_loader=dict(
#         batch_size=1,
#         serial_batches=True,
#         num_threads=0,
#         pin_memory=False,
#         collator=[
#             dict(type='ListCollateFN', names=('poly', 'poly_meta')),
#         ]
#     ),
#     dataset=dict(
#         max_size=-1,
#         reader=dict(type='ICDARDetReader', train=True, root='../datasets/ICDAR2015', use_pil=False),
#         internodes=[
#             dict(type='DataSource'),
#             dict(type='ResizeAndPadding', 
#                 resize=dict(
#                     type='Resize',
#                     size=(960, 960),
#                     keep_ratio=True,
#                     short=False,
#                 ),
#                 padding=dict(
#                     type='PaddingBySize',
#                     size=(960, 960),
#                     fill=(0, 0, 0), 
#                     padding_mode='constant',
#                     center=False,
#                     one_way='forward'
#                 ),
#                 # one_way='forward'
#             ),
#             dict(type='ToPILImage'),
#             dict(type='ToTensor'),
#         ],
#     ),
# )

