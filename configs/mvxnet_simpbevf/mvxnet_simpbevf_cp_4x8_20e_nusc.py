_base_ = [
    '../_base_/datasets/nusc_cp.py',
    '../_base_/models/centerpoint_01voxel_second_secfpn_nus.py',
    '../_base_/schedules/cyclic_20e.py',
    '../_base_/default_runtime.py'
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
    lc_fusion=True,
    freeze_img=True,
    grid=0.6, # 0.075*8
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
    pts_voxel_layer=dict(voxel_size=voxel_size, point_cloud_range=point_cloud_range),
    pts_middle_encoder=dict(sparse_shape=[41, 1440, 1440]),
    pts_bbox_head=dict(
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
freeze_lidar_components = False
find_unused_parameters = True
no_freeze_head = True

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,)

optimizer = dict(lr=5e-5)
evaluation = dict(interval=5)
total_epochs = 20
# load_lift_from = 'work_dirs/bevf_cp_4x8_20e_nusc_cam/epoch_20.pth'
# https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/mask_rcnn_r50_fpn_coco-2x_1x_nuim/mask_rcnn_r50_fpn_coco-2x_1x_nuim_20201008_195238-b1742a60.pth
# load_img_from = 'checkpoints/mask_rcnn_r50_fpn_coco-2x_1x_nuim_20201008_195238-b1742a60.pth'
# https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20210827_161135-1782af3e.pth
# load_from = 'checkpoints/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20210827_161135-1782af3e.pth'
load_from = 'checkpoints/mask_rcnn_r50_fpn_plus_centerpoint_wo_head.pth'

# gpu_ids = [0]
