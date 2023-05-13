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
            dict(type='ListCollateFN', names=('image_meta', 'bbox','bbox_meta', 'point', 'point_meta', 'tsr_l_row', 'tsr_c_row', 'tsr_l_col', 'tsr_c_col')),
            # dict(type='ListCollateFN', names=('image_meta', 'point_meta')),
        ]
    ),
    dataset=dict(
        reader=dict(type='WTWReader', root='../datasets/wtw/test'),
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
            # dict(type='CalcTSRGT'),
            dict(type='ToTensor'),
            # dict(type='CalcCenterNetGrids', ratio=0.25, num_classes=1, use_point=True),
            # dict(type='CalcPTSGrids', ratio=1),
            dict(type='CalcTSRGT'),
        ],
    ),
)
