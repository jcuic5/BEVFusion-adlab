_base_ = [
    '../../_base_/datasets/nusc_cam_cp.py',
    '../../_base_/models/centerpoint_dcn_nus.py',
    '../../_base_/schedules/cyclic_20e.py', 
    '../../_base_/default_runtime.py'
]
voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54, -54, -5.0, 54, 54, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

final_dim=(900, 1600) # HxW
downsample=8
imc = 256
model = dict(
    type='MVXSimpleBEVFusionCP',
    lss=True,
    se=False,
    camera_stream=True, 
    lc_fusion=False,
    grid=0.6, 
    num_views=6,
    final_dim=final_dim,
    downsample=downsample, 
    imc=imc, 
    lic=256 * 2,
    pc_range = point_cloud_range,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=imc,
        separate_head=dict(
            type='DCNSeparateHead',
            dcn_config=dict(
                type='DCN',
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                groups=4),
            init_bias=-2.19,
            final_kernel=3),
        bbox_coder=dict(
            voxel_size=voxel_size[:2], pc_range=point_cloud_range[:2])),
    train_cfg=dict(
        pts=dict(
            grid_size=[1440, 1440, 40],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(voxel_size=voxel_size[:2], pc_range=point_cloud_range[:2], nms_type='circle')))


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=6,)
# https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/mask_rcnn_r50_fpn_coco-2x_1x_nuim/mask_rcnn_r50_fpn_coco-2x_1x_nuim_20201008_195238-b1742a60.pth
load_img_from = 'checkpoints/mask_rcnn_r50_fpn_coco-2x_1x_nuim_20201008_195238-b1742a60.pth'
