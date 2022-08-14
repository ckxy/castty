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
        batch_size=2,
        serial_batches=True,
        num_threads=0,
        pin_memory=False,
        collator=[
            dict(type='LabelCollateFN', names=('label',)),
        ]
    ),
    dataset=dict(
        reader=dict(type='ImageFolderReader', root='../datasets/kagglecatsanddogs_3367a/PetImages'),
        # reader=dict(type='ImageFolderReader', root='../datasets/tiny-imagenet-200/train'),
        # reader=dict(type='DukeMTMCAttritubesReader', root='../datasets/DukeMTMC-reID', group='train', mode='ab'),
        internodes=[
            dict(type='DataSource'),
            # dict(type='CutMix', internodes=[
            #     dict(type='DataSource'),
            #     dict(type='Resize', size=(640, 640), keep_ratio=True, short=False),
            # ]),
            dict(type='Resize', size=(640, 640), keep_ratio=False),
            dict(type='CenterCrop', size=(480, 480)),
            # dict(type='ToPILImage'),
            # dict(type='OneHotEncode', num_classes=2),
            dict(type='ToTensor'),
        ],
    ),
)

