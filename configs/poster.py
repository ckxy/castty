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
            # dict(type='ListCollateFN', names=('image_meta', 'remove_meta')),
            dict(type='ListCollateFN', names=('image_meta',)),
            dict(type='ListCollateFN', names=('image',)),
            dict(type='BboxCollateFN', names=('bbox',)),
            dict(type='ListCollateFN', names=('bbox_meta',)),
        ]
    ),
    dataset=dict(
        # reader=dict(type='CanvasLayoutReader', use_pil=True, root='../datasets/PKU_PosterLayout_Annotations/train', csv_path='../datasets/PKU_PosterLayout_Annotations/train_csv_9973.csv'),
        reader=dict(type='PSDParseReader', use_pil=True, root='../datasets/psddata/res'),
        internodes=[
            dict(type='DataSource'),
            # dict(type='ResizeAndPadding', resize=dict(type='Resize', size=(240, 350), keep_ratio=False)),
            # dict(type='Resize', size=(1024, 1024), keep_ratio=True, short=False),
            # dict(type='ToTensor'),
            # dict(type='CalcDSLabel'),
            # dict(type='DSMerge'),
        ],
    ),
)

