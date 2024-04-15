import os

config_global = "sherlock"  # "local" or "sherlock

constants = {}
if config_global == "local":
    constants = {
        "mmposeDirectory": "/home/yoni/OneDrive_yonigoz@stanford.edu/RA/Code/mmpose",
        "model_config_person": "demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py",
        "model_ckpt_person": "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
        # "model_config_person" : "demo/mmdetection_cfg/configs/convnext/cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py",
        # "model_ckpt_person" :"https://download.openmmlab.com/mmdetection/v2.0/convnext/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220509_204200-8f07c40b.pth",
        "model_config_pose": "configs/body_2d_keypoint/topdown_heatmap/infinity/td-hm_hrnet-w48_dark-8xb32-210e_merge_bedlam_infinity_coco_eval_bedlam-384x288_pretrained.py",
        "model_ckpt_pose": "pretrain/hrnet/best_infinity_AP_epoch_21.pth",
        "dataDir": "/home/yoni/OneDrive_yonigoz@stanford.edu/RA/Code/OpenCap/data",
        "batch_size_det": 4,
        "batch_size_pose": 8,
    }
    constants["model_ckpt_pose_absolute"] = os.path.join(
        constants["mmposeDirectory"], constants["model_ckpt_pose"]
    )

if config_global == "windows":
    constants = {
        "mmposeDirectory": "C:/Data/OpenCap/mmpose",
        "model_config_person": "demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py",
        "model_ckpt_person": "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
        # "model_config_person" : "demo/mmdetection_cfg/configs/convnext/cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py",
        # "model_ckpt_person" :"https://download.openmmlab.com/mmdetection/v2.0/convnext/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220509_204200-8f07c40b.pth",
        "model_config_pose": "configs/body_2d_keypoint/topdown_heatmap/infinity/td-hm_hrnet-w48_dark-8xb32-210e_merge_bedlam_infinity_coco_eval_bedlam-384x288_pretrained.py",
        "model_ckpt_pose": "pretrain/hrnet/best_infinity_AP_epoch_27.pth",
        "dataDir": "C:/Data/OpenCap/data",
        "batch_size_det": 4,
        "batch_size_pose": 8,
    }
    constants["model_ckpt_pose_absolute"] = os.path.join(
        constants["mmposeDirectory"], constants["model_ckpt_pose"]
    )


if config_global == "sherlock":
    constants = {
        "mmposeDirectory": "/home/users/yonigoz/RA/mmpose",
        "model_config_person": "demo/mmdetection_cfg/configs/convnext/cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py",
        "model_ckpt_person": "https://download.openmmlab.com/mmdetection/v2.0/convnext/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220509_204200-8f07c40b.pth",
        # "model_config_person" : "demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py",
        # "model_ckpt_person" : "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
        # "model_config_pose" : "configs/body_2d_keypoint/topdown_heatmap/infinity/td-hm_ViTPose-huge_8xb64-210e_merge_bedlam_infinity_eval_bedlam-256x192.py",
        # "model_ckpt_pose" : "/scratch/users/yonigoz/mmpose_data/work_dirs/merge_bedlam_infinity_eval_bedlam/ViT/huge/best_infinity_AP_epoch_10.pth",
        "model_ckpt_pose": "/scratch/users/yonigoz/mmpose_data/work_dirs/merge_bedlam_infinity_coco_eval_bedlam/HRNet/w48_dark_pretrained/best_infinity_AP_epoch_27.pth",
        "dataDir": "/scratch/users/yonigoz/OpenCap_data",
        "batch_size_det": 32,
        "batch_size_pose": 4,
    }
    constants["model_ckpt_pose_absolute"] = constants["model_ckpt_pose"]


def getMMposeAnatomicalCocoMarkerNames():
    marker_names = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "sternum",
        "rshoulder",
        "lshoulder",
        "r_lelbow",
        "l_lelbow",
        "r_melbow",
        "l_melbow",
        "r_lwrist",
        "l_lwrist",
        "r_mwrist",
        "l_mwrist",
        "r_ASIS",
        "l_ASIS",
        "r_PSIS",
        "l_PSIS",
        "r_knee",
        "l_knee",
        "r_mknee",
        "l_mknee",
        "r_ankle",
        "l_ankle",
        "r_mankle",
        "l_mankle",
        "r_5meta",
        "l_5meta",
        "r_toe",
        "l_toe",
        "r_big_toe",
        "l_big_toe",
        "l_calc",
        "r_calc",
        "C7",
        "L2",
        "T11",
        "T6",
    ]

    return marker_names


def getMMposeMarkerNames():
    markerNames = [
        "Nose",
        "LEye",
        "REye",
        "LEar",
        "REar",
        "LShoulder",
        "RShoulder",
        "LElbow",
        "RElbow",
        "LWrist",
        "RWrist",
        "LHip",
        "RHip",
        "LKnee",
        "RKnee",
        "LAnkle",
        "RAnkle",
        "LBigToe",
        "LSmallToe",
        "LHeel",
        "RBigToe",
        "RSmallToe",
        "RHeel",
    ]

    return markerNames


def getOpenPoseMarkerNames():
    markerNames = [
        "Nose",
        "Neck",
        "RShoulder",
        "RElbow",
        "RWrist",
        "LShoulder",
        "LElbow",
        "LWrist",
        "midHip",
        "RHip",
        "RKnee",
        "RAnkle",
        "LHip",
        "LKnee",
        "LAnkle",
        "REye",
        "LEye",
        "REar",
        "LEar",
        "LBigToe",
        "LSmallToe",
        "LHeel",
        "RBigToe",
        "RSmallToe",
        "RHeel",
    ]

    return markerNames


def getMMposeAnatomicalCocoMarkerPairs():
    markerNames = {
        "left_eye": "right_eye",
        "left_ear": "right_ear",
        "left_shoulder": "right_shoulder",
        "left_elbow": "right_elbow",
        "left_wrist": "right_wrist",
        "left_hip": "right_hip",
        "left_knee": "right_knee",
        "left_ankle": "right_ankle",
        "rshoulder": "lshoulder",
        "lshoulder": "rshoulder",
        "r_lelbow": "l_lelbow",
        "l_lelbow": "r_lelbow",
        "r_melbow": "l_melbow",
        "l_melbow": "r_melbow",
        "r_lwrist": "l_lwrist",
        "l_lwrist": "r_lwrist",
        "r_mwrist": "l_mwrist",
        "l_mwrist": "r_mwrist",
        "r_ASIS": "l_ASIS",
        "l_ASIS": "r_ASIS",
        "r_PSIS": "l_PSIS",
        "l_PSIS": "r_PSIS",
        "r_knee": "l_knee",
        "l_knee": "r_knee",
        "r_mknee": "l_mknee",
        "l_mknee": "r_mknee",
        "r_ankle": "l_ankle",
        "l_ankle": "r_ankle",
        "r_mankle": "l_mankle",
        "l_mankle": "r_mankle",
        "r_5meta": "l_5meta",
        "l_5meta": "r_5meta",
        "r_toe": "l_toe",
        "l_toe": "r_toe",
        "r_big_toe": "l_big_toe",
        "l_big_toe": "r_big_toe",
        "l_calc": "r_calc",
        "r_calc": "l_calc",
    }

    return markerNames
