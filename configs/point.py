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
            # dict(type='BboxCollateFN', names=('bbox',)),
            dict(type='ListCollateFN', names=('point_meta',)),
        ]
    ),
    dataset=dict(
        # reader=dict(type='MPIIReader', root='../datasets/mpii', set_path='../datasets/mpii/mpii_human_pose_v1_u12_1.mat'),
        # reader=dict(type='LSPReader', root='../datasets/lsp', set_path='../datasets/lsp/LEEDS_annotations.json', is_test=False),
        reader=dict(type='WFLWReader', root='../datasets/WFLW/WFLW', txt_path='WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt', use_pil=True),
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
            # dict(type='CopyTag', src_tag='image', dst_tag='ori_image'),
            # dict(type='ToCV2Image'),
            # dict(type='AdaptiveCrop'),
            # dict(type='Padding', padding=(20, 30, 40, 50), fill=50, padding_mode='reflect'),
            # dict(type='PaddingBySize', size=(416, 416), fill=50, padding_mode='constant', center=True),
            # dict(type='PaddingByStride', stride=32, fill=(50, 50, 50), padding_mode='constant', center=False, one_way='orward'),
            # dict(type='Resize', size=(512, 512), keep_ratio=True, short=False),
            # dict(type='PaddingBySize', size=(512, 512), padding_mode='constant', center=False),
            # dict(type='WarpResize', size=(416, 416), expand=True, keep_ratio=True, short=False, one_way='forward'),
            # dict(type='ResizeAndPadding', 
            #     resize=dict(
            #         type='Resize',
            #         size=(416, 416),
            #         keep_ratio=True,
            #         short=False,
            #     ),
            #     padding=dict(
            #         type='PaddingBySize',
            #         size=(1600, 1800),
            #         # type='PaddingByStride',
            #         # stride=32,
            #         fill=(0, 0, 0), 
            #         padding_mode='constant',
            #         center=False
            #     ),
            #     # one_way='forward'
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
            dict(type='WFLWCrop'),
            # dict(type='AdaptiveTranslate'),
            # dict(type='MinIOGCrop', threshs=(-1, 0.1, 0.3, 0.5, 0.7, 0.9)),
            # dict(type='GridMask', use_w=True, use_h=True, rotate=0, offset=False, invert=False, ratio=0.5),
            # dict(type='ToPILImage'),
            # dict(type='CalcHeatmapByPoint', ratio=0.25),
            dict(type='ToTensor'),
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
            dict(type='ListCollateFN', names=('bbox','bbox_meta', 'point', 'point_meta', 'tsr_l_row', 'tsr_c_row', 'tsr_l_col', 'tsr_c_col')),
            # dict(type='ListCollateFN', names=('point', 'point_meta')),
        ]
    ),
    dataset=dict(
        reader=dict(type='WTWSTReader', root='../datasets/wtw/test'),
        # reader=dict(type='COCOAPIReader', use_pil=True, use_keypoint=True, set_path='../datasets/coco/annotations/person_keypoints_val2017.json', img_root='../datasets/coco/val2017'),
        internodes=[
            dict(type='DataSource'),
            # dict(type='RandomCenterCropPad', size=(512, 512), ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3)),
            # dict(type='ResizeAndPadding', 
            #     resize=dict(
            #         type='WarpResize',
            #         size=(504, 504),
            #         keep_ratio=True,
            #         short=False,
            #     ),
            #     padding=dict(
            #         type='PaddingBySize',
            #         size=(512, 512),
            #         fill=(0, 0, 0), 
            #         padding_mode='constant',
            #         center=True
            #     ),
            #     # one_way='forward'
            # ),
            # dict(type='ToPILImage'),
            dict(type='CalcTSRGT'),
            dict(type='ToTensor'),
            # dict(type='CalcCenterNetGrids', ratio=0.25, num_classes=1, use_point=True),
            # dict(type='CalcPTSGrids', ratio=0.25),
            # dict(type='CalcTSRGT'),
        ],
    ),
)
