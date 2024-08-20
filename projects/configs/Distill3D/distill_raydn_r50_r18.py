_base_ = [
    "../../../mmdetection3d/configs/_base_/datasets/nus-3d.py",
    "../../../mmdetection3d/configs/_base_/default_runtime.py",
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# knowledge distillation settings
work_dir = "work_dirs/distill_raydn_r50-18_bkl_24e_4x4_lr4e-4"

teacher_cfg = "projects/configs/Distill3D/raydn_r50_704_bs2_seq_428q_nui_24e.py"
student_cfg = "projects/configs/Distill3D/raydn_r18_704_bs2_seq_428q_nui_24e.py"
# dist_params = dict(backend='nccl') #, port=29501)

num_gpus = 4
batch_size = 4
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
num_epochs = 24

queue_length = 1  # each sequence contains `queue_length` frames.
num_frame_losses = 1
collect_keys=['lidar2img', 'intrinsics', 'extrinsics','timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

## FPN distillation settings
weight_fpn = 0.1
reverse_fpn = 7.0 
temp_stu_fpn = 4.0
temp_tea_fpn = 4.0
## Transfomer FPN distillation settings
weight_tfpn = 0.1
reverse_tfpn = 7.0 
temp_stu_tfpn = 2.0
temp_tea_tfpn = 2.0
## Attention*TFPN/FPN distillation settings
weight_attn_bdd = 0.1
weight_attn_l2 = 0.1
reverse_attn = 7.0
temp_stu_attn = 2.0
temp_tea_attn = 2.0
## Attention*BEV distillation settings
weight_bevattn = 0.1
reverse_bevattn = 7.0
temp_stu_bevattn = 4.0
temp_tea_bevattn = 4.0

distiller = dict(
    type="HarmoDistiller",
    teacher_pretrained="ckpts/raydn_r50_704_bs2_seq_428q_nui_60e_20240403.pth",
    distill_on_FPN=True, #True,
    distill_on_TFPN=False,
    distill_on_ATTN=False,
    distill_on_DN=False,
    distill_on_BEVATTN=False,
    duplicate_student_head=False,
    distill_on_outputs_w=0, ## output query distill
    update_by_iter=True,
    distill_on_diff_dim=False,
    distiller_fpn=[
        dict(
            student_module="img_neck.fpn_convs.0.conv",
            teacher_module="img_neck.fpn_convs.0.conv",
            output_hook=True,
            methods=[
                dict(
                    type="BalanceKLDivergence",
                    temp_stu=temp_stu_fpn,
                    temp_tea=temp_tea_fpn,
                    name="loss_bkl_fpn_0",
                    student_channels=256,
                    teacher_channels=256,
                    weight=weight_fpn,
                    reverse=reverse_fpn,
                )
            ],
        ),
        dict(
            student_module="img_neck.fpn_convs.1.conv",
            teacher_module="img_neck.fpn_convs.1.conv",
            output_hook=True,
            methods=[
                dict(
                    type="BalanceKLDivergence",
                    temp_stu=temp_stu_fpn,
                    temp_tea=temp_tea_fpn,
                    name="loss_bkl_fpn_1",
                    student_channels=256,
                    teacher_channels=256,
                    weight=weight_fpn,
                    reverse=reverse_fpn,
                )
            ],
        ),
        dict(
            student_module="img_neck.fpn_convs.2.conv",
            teacher_module="img_neck.fpn_convs.2.conv",
            output_hook=True,
            methods=[
                dict(
                    type="BalanceKLDivergence",
                    temp_stu=temp_stu_fpn,
                    temp_tea=temp_tea_fpn,
                    name="loss_bkl_fpn_2",
                    student_channels=256,
                    teacher_channels=256,
                    weight=weight_fpn,
                    reverse=reverse_fpn,
                )
            ],
        ),
        dict(
            student_module="img_neck.fpn_convs.3.conv",
            teacher_module="img_neck.fpn_convs.3.conv",
            output_hook=True,
            methods=[
                dict(
                    type="BalanceKLDivergence",
                    temp_stu=temp_stu_fpn,
                    temp_tea=temp_tea_fpn,
                    name="loss_bkl_fpn_3",
                    student_channels=256,
                    teacher_channels=256,
                    weight=weight_fpn*0.1,
                    reverse=reverse_fpn,
                )
            ],
        ),
    ],
)

dataset_type = "CustomNuScenesDataset"
data_root = "data/nuscenes/"
file_client_args = dict(backend="disk")

ida_aug_conf = {
        "resize_lim": (0.38, 0.55),
        "final_dim": (256, 704),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": True,
    }
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
        with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipRotImage', data_aug_conf = ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage',
            rot_range=[-0.3925, 0.3925],
            translation_std=[0, 0, 0],
            scale_ratio_range=[0.95, 1.05],
            reverse_angle=True,
            training=True,
            ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'prev_exists'] + collect_keys,
             meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d','gt_labels_3d'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipRotImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys,
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'] + collect_keys,
            meta_keys=('filename', 'ori_shape', 'img_shape','pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token'))
        ])
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes2d_temporal_infos_train.pkl',
        num_frame_losses=num_frame_losses,
        seq_split_num=2, # streaming video training
        seq_mode=True, # streaming video training
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=False,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type, pipeline=test_pipeline, collect_keys=collect_keys + ['img', 'img_metas'], queue_length=queue_length, ann_file=data_root + 'nuscenes2d_temporal_infos_val.pkl', classes=class_names, modality=input_modality),
    test=dict(type=dataset_type, pipeline=test_pipeline, collect_keys=collect_keys + ['img', 'img_metas'], queue_length=queue_length, ann_file=data_root + 'nuscenes2d_temporal_infos_val.pkl', classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
    )

optimizer = dict(
    type='AdamW',
    lr=4e-4, # bs 8: 2e-4 || bs 16: 4e-4
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.25), # 0.25 only for Focal-PETR with R50-in1k pretrained weights
        }),
    weight_decay=0.01)

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic', grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

evaluation = dict(interval=num_iters_per_epoch * num_epochs, pipeline=test_pipeline)
find_unused_parameters=False #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
runner = dict(type="CustomerIterBasedRunner", max_iters=num_epochs * num_iters_per_epoch)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='WandbLoggerHook',
            by_epoch=False,
            init_kwargs=dict(
                project="HarmoDistill_r18_adl",
                dir=work_dir,
                name=work_dir,
            ),
        ),
    ])
