_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/dior.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# please install mmcls>=0.22.0
# import mmcls.models to trigger register_module in mmcls
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmcls.ConvNeXt',
        arch='tiny',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(in_channels=[96, 192, 384, 768]),
    roi_head=dict(
        mask_roi_extractor=None,
        mask_head=None,
        bbox_head=dict(num_classes=20)
    )
)

# optimizer
optimizer = dict(
    _delete_=True,
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={
        'decay_rate': 0.95,
        'decay_type': 'layer_wise',
        'num_layers': 6
    })
lr_config = dict(warmup_iters=1000, step=[100])
runner = dict(type='EpochBasedRunner', max_epochs=120)

evaluation = dict(interval=12, metric=['mAP'])
checkpoint_config = dict(interval=12)

init_kwargs = {
    'project': 'rsi-detection',
    'entity': 'tum-tanmlh',
    'name': 'mask_rcnn_convnext-t_p4_w7_fpn_120e_dior',
    'resume': 'never'
}
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        dict(type='MMDetWandbHook',
             init_kwargs=init_kwargs,
             interval=10,
             log_checkpoint=True,
             log_checkpoint_metadata=True,
             num_eval_images=30,
             bbox_score_thr=0.3,
             eval_after_run=True)
    ])
