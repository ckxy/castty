general = dict(
    experiment_name='vis_test',
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
            dict(type='ListCollateFN', names=('image_meta', 'poly', 'poly_meta')),
        ]
    ),
    dataset=dict(
        # reader=dict(type='ICDARDetReader', root='../datasets/ICDAR2015', use_pil=False),
        reader=dict(type='DDI100SubsetDetReader', root='/Users/liaya/Documents/datasets/ddi_100/17'),
        internodes=[
            dict(type='DataSource'),
            # dict(type='Mosaic', internodes=[
            #     dict(type='DataSource'),
            # ]),
            # dict(type='EastRandomCrop'),
            # dict(type='ResizeAndPadding', 
            #     resize=dict(
            #         type='Resize',
            #         size=(960, 960),
            #         keep_ratio=True,
            #         short=False,
            #     ),
            #     padding=dict(
            #         type='PaddingBySize',
            #         size=(960, 960),
            #         fill=(0, 0, 0), 
            #         padding_mode='constant',
            #     ),
            #     one_way='forward'
            # ),
            # dict(type='Flip', horizontal=True, p=0.5),
            # dict(type='WarpRotate', angle=(-10, 10), expand=True),
            # dict(type='Rescale', ratio_range=(0.5, 3), mode='range'),
            # dict(type='RescaleLimitedByBound', long_size_bound=1280, short_size_bound=640, ratio_range=(0.5, 3), mode='range'),
            # dict(type='WarpPerspective', distortion_scale=1, ccs=True),
            # dict(type='FilterSelfOverlapping'),
            # dict(type='AdaptiveCrop'),
            # dict(type='AdaptiveTranslate'),
            # dict(type='PSEEncode', num_classes=2),
            # dict(type='DBEncode', num_classes=2),
            # dict(type='EraseTags', tags=['poly', 'poly_meta']),
            # dict(type='PSECrop', size=(640, 640), positive_sample_ratio=5.0 / 8.0),
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
#         # sampler=dict(type='CustomWeightedRandomSampler', weights=(1, 2), num_samples=100),
#         # batch_sampler=dict(type='StepsBatchSampler', steps=100),
#         collator=[
#             dict(type='ListCollateFN', names=('image_meta', 'poly', 'poly_meta')),
#         ]
#     ),
#     dataset=dict(
#         # reader=dict(type='ICDARDetReader', root='../datasets/ICDAR2015', use_pil=True),
#         reader=dict(type='WTWLineReader', root='../datasets/wtw/test'),
#         # reader=dict(
#         #     type='CatReader', 
#         #     readers=(
#         #         dict(type='ICDARDetReader', root='../datasets/ICDAR2015'),
#         #         dict(type='ICDARDetReader', root='../datasets/ICDAR2015', train=False),
#         #     ),
#         #     use_pil=False,
#         # ),
#         internodes=[
#             # dict(type='ChooseOne', branchs=[
#             #     dict(type='Mosaic', internodes=[
#             #         dict(type='DataSource'),
#             #         dict(type='EastRandomCrop', min_crop_side_ratio=0.4),
#             #         dict(type='Resize', size=(640, 640), keep_ratio=True, short=False),
#             #         dict(type='Rot90', k=[1, 2, 3], p=0.5),
#             #     ]),
#             #     dict(type='Bamboo', internodes=[
#             #         dict(type='DataSource'),
#             #         dict(type='EastRandomCrop', min_crop_side_ratio=0.4),
#             #         dict(type='Rot90', k=[1, 2, 3], p=0.5),
#             #         dict(type='RandomExpand', ratio=2, p=0.5),
#             #     ]),
#             # ]),
#             # dict(type='Mosaic', internodes=[
#             #     dict(type='DataSource'),
#             #     dict(type='EastRandomCrop', min_crop_side_ratio=0.4),
#             #     dict(type='Resize', size=(640, 640), keep_ratio=True, short=False),
#             #     dict(type='Rot90', k=[1, 2, 3], p=0.5),
#             # ]),
#             dict(type='DataSource'),
#             # dict(type='EastRandomCrop'),
#             # dict(type='Rot90', k=[1, 2, 3]),
#             # dict(type='WarpRotate', angle=(-30, 0), expand=True),
#             # dict(type='Resize', size=(1280, 1280), keep_ratio=True, short=False),
#             # dict(type='PaddingBySize', size=(1280, 1280), fill=(0, 0, 0), padding_mode='constant', center=False),
#             # dict(type='ToPILImage'),
#             dict(type='CalcLinkMap'),
#             # dict(type='DBEncode'),
#             # dict(type='EraseTags', tags=['poly', 'poly_meta']),
#             # dict(type='PSEEncode'),
#             # dict(type='PSECrop', size=(640, 640), positive_sample_ratio=5.0 / 8.0),
#             dict(type='ToTensor'),
#             # dict(type='DBMCEncode', num_classes=2),
#         ],
#     ),
# )


