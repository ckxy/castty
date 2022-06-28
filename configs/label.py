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
        # collator=[
        #     dict(type='ListCollateFN', names=('poly', 'poly_meta')),
        # ]
    ),
    dataset=dict(
        reader=dict(type='ImageFolderReader', root='../datasets/kagglecatsanddogs_3367a/PetImages'),
        internodes=[
            dict(type='DataSource'),
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
            # dict(type='RescaleLimitedByBound', long_size_bound=1280, short_size_bound=640, ratio_range=(0.5, 3), mode='range'),
            # dict(type='ToPILImage'),
            dict(type='OneHotEncode', num_classes=2),
            dict(type='ToTensor'),
        ],
    ),
)

