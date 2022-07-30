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
        serial_batches=False,
        num_threads=0,
        pin_memory=False,
        # collator=[
        #     dict(type='ListCollateFN', names=('label_meta',)),
        # ]
    ),
    dataset=dict(
        reader=dict(type='ImageFolderReader', root='../datasets/kagglecatsanddogs_3367a/PetImages'),
        # reader=dict(type='ImageFolderReader', root='../datasets/tiny-imagenet-200/train'),
        # reader=dict(type='DukeMTMCAttritubesReader', root='../datasets/DukeMTMC-reID', group='train', mode='b'),
        internodes=[
            # dict(type='DataSource'),
            dict(type='MixUp', internodes=[
                dict(type='DataSource'),
                dict(type='Resize', size=(640, 640), keep_ratio=True, short=False),
            ]),
            # dict(type='ToPILImage'),
            # dict(type='OneHotEncode', num_classes=2),
            dict(type='ToTensor'),
        ],
    ),
)

