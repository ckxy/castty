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
            dict(type='ListCollateFN', names=('image_meta',)),
            dict(type='BboxCollateFN', names=('bbox',)),
            dict(type='ListCollateFN', names=('bbox_meta',)),
            dict(type='ListCollateFN', names=('point_meta',)),
        ]
    ),
    dataset=dict(
        reader=dict(type='CCPDFolderReader', use_pil=True, root='../datasets/ccpd/CCPD2020/ccpd_green/train'),
        internodes=[
            dict(type='DataSource'),
            dict(type='ToTensor'),
        ],
    ),
)
