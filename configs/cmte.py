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
            dict(type='ListCollateFN', names=('a_image_meta', 'a_star_image', 'b_image_meta')),
            dict(type='LabelCollateFN', names=('a_label', 'b_label')),
        ]
    ),
    dataset=dict(
        reader=dict(type='C2NReader', root='../Controllable_Multi-Textures_Expansion/datasets/half/mix3', use_pil=True),
        # tag_mapping=dict(image=['image', 'aimg']),
        internodes=[
            dict(type='DataSource'),
            dict(type='Flip', horizontal=True),
            dict(type='Crop', size=(256, 256), tag_mapping=dict(image=['image', 'b_image'])),
            dict(type='Crop', size=(128, 128), tag_mapping=dict(image=['image'])),
            dict(type='RenameTag', old_name='image', new_name='a_image'),
            dict(type='RenameTag', old_name='a_star_image', new_name='image'),
            dict(type='Resize', size=(128, 128), keep_ratio=True, short=True, tag_mapping=dict(image=['image'])),
            dict(type='Crop', size=(128, 128), tag_mapping=dict(image=['image'])),
            dict(type='RenameTag', old_name='image', new_name='a_star_image'),
            dict(type='RenameTag', old_name='a_image', new_name='image'),
            dict(type='ToTensor'),
            dict(type='RenameTag', old_name='image', new_name='a_image'),
            dict(type='RenameTag', old_name='image_meta', new_name='a_image_meta'),
            dict(type='RenameTag', old_name='label', new_name='a_label'),
        ],
    ),
)
