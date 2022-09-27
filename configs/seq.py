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
            dict(type='EnSeqCollateFN', names=('encoded_seq',)),
        ]
    ),
    dataset=dict(
        reader=dict(type='LmdbDTRBDataset', use_pil=False, root='../datasets/deep-text/evaluation/IC15_1811', char_path='../datasets/deep-text/character.txt'),
        internodes=[
            dict(type='DataSource'),
            dict(type='Resize', size=(320, 32), keep_ratio=True),
            dict(type='PaddingBySize', size=(320, 32), fill=(0, 0, 0), padding_mode='constant', center=False),
            dict(type='ToGrayscale'),
            dict(type='ToPILImage'),
            dict(type='ToTensor'),
            dict(type='CTCEncode', char_path='../datasets/deep-text/character.txt'),
            dict(type='To1CHTensor'),
        ],
    ),
)

