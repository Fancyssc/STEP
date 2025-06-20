# Copyright (c) OpenMMLab. All rights reserved.
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='VGGNet',
        depth=16,
        with_bn=True,
        num_stages=5,
        out_indices=(0, 1, 2, 3, 4),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://vgg16_bn')
    ),
    decode_head=dict(
        type='SegNetHead',
        in_channels=512,
        channels=512,
        num_convs=5,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=12,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)