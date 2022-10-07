_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/dior.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='MaskRCNN',
    roi_head=dict(
        mask_roi_extractor=None,
        mask_head=None,
        bbox_head=dict(num_classes=20)
    )
)

# optimizer
optimizer = dict(
    # _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
optimizer_config = dict(grad_clip=None)


# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,  # same as burn-in in darknet
    warmup_ratio=0.1,
    step=[100])

runner = dict(type='EpochBasedRunner', max_epochs=120)


evaluation = dict(interval=12, metric=['mAP'])
checkpoint_config = dict(interval=12)

init_kwargs = {
    'project': 'rsi-detection',
    'entity': 'tum-tanmlh',
    'name': 'mask_rcnn_r50_fpn_120e_dior',
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
             bbox_score_thr=0.3)
    ])
