import glob
import os
import os.path as osp
import zipfile
from typing import Optional

import cv2
import numpy as np
from constants import JOINT_NAMES, K1, K2
from pycocotools.coco import COCO
from scipy.spatial.transform import Rotation


class VideoScene:
    def __init__(
        self,
        path_to_example: Optional[str] = None,
        exercise: Optional[str] = None,
        index_video: Optional[int] = None,
        video_name="video",
        labels_name="labels",
        segmentation_name="segmentation",
    ):
        if path_to_example is None:
            self.json_path = (
                f"synthetic_finetuning/data/api_example/{exercise}/{labels_name}.json"
            )
            self.video_path = (
                f"synthetic_finetuning/data/api_example/{exercise}/{video_name}.mp4"
            )
            iseg_paths = sorted(
                glob.glob(
                    osp.join(
                        f"synthetic_finetuning/data/api_example/{exercise}/{segmentation_name}",
                        "*.iseg.*.png",
                    )
                )
            )

            nb_elements = len(
                glob.glob(
                    osp.join(
                        f"synthetic_finetuning/data/api_example/{exercise}/{segmentation_name}",
                        "image.000000.iseg.*.png",
                    )
                )
            )

        else:
            self.json_path = osp.join(path_to_example, f"{labels_name}.json")
            self.video_path = osp.join(path_to_example, f"{video_name}.mp4")
            iseg_paths_glob = osp.join(
                path_to_example, f"{segmentation_name}/*.iseg.*.png"
            )
            iseg_path = osp.join(path_to_example, segmentation_name)
            # check if path exists
            if not sorted(glob.glob(iseg_paths_glob)):
                # create directory
                os.makedirs(iseg_path)
                with zipfile.ZipFile(f"{iseg_path}.zip", "r") as zip_ref:
                    zip_ref.extractall(iseg_path)

            iseg_paths = sorted(glob.glob(iseg_paths_glob))

            iseg_elements_glob = osp.join(
                path_to_example, f"{segmentation_name}/image.000000.iseg.*.png"
            )
            nb_elements = len(glob.glob(iseg_elements_glob))
            if not iseg_paths:
                archive = zipfile.ZipFile(f"{path_to_example}.zip", "r")
                imgdata = archive.read("img_01.png")

        self.iseg_paths = [
            path for path in iseg_paths if int(path.split(".")[-2]) % nb_elements == 0
        ]

        self.coco = COCO(self.json_path)
        self.infos = self.coco.dataset.get("info")
        self.scene_id = self.infos["scene_id"]
        if "camera_K_matrix" in self.infos:
            self.K = np.array(self.infos["camera_K_matrix"])
        else:
            self.K = K2 if self.scene_id == "4578713" else K1
        if "camera_RT_matrix" in self.infos:
            self.current_extrinsic_matrix = np.array(self.infos["camera_RT_matrix"])
        self.current_img = None
        self.current_ann = None
        self.current_seg = None
        self.current_vertices = None

        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.nb_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.dims = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.mean_accuracy = []

    def load_frame(self, index_frame: int):
        img_data = list(self.coco.imgs.values())[index_frame]
        img_id = img_data["id"]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index_frame)
        res, img = self.cap.read()
        if not res:
            img = None

        anns = [ann for ann in anns if "armature_keypoints" in ann]
        if len(anns) > 1:
            print("Multiple humans detected")
        ann = anns[0]

        self.current_img = img
        self.current_ann = ann
        self.current_seg = cv2.imread(self.iseg_paths[index_frame])

        return img, ann, self.infos

    def compute_extrinsic_matrix(self, joints: np.ndarray):
        image_points = np.array(
            [
                [
                    self.current_ann["armature_keypoints"][joint_name]["x"],
                    self.current_ann["armature_keypoints"][joint_name]["y"],
                ]
                for joint_name in JOINT_NAMES[:55]
            ],
            dtype=np.float32,
        )

        _, rvec, tvec = cv2.solvePnP(joints[:55], image_points, self.K, None)
        R, _ = cv2.Rodrigues(rvec)
        T = np.concatenate([R, tvec], axis=1)

        self.current_extrinsic_matrix = T

    def align_3d_vertices(self, joints: np.ndarray, augmented_vertices: np.ndarray):
        image_points = np.array(
            [
                [
                    self.current_ann["armature_keypoints"][joint_name]["x_global"],
                    self.current_ann["armature_keypoints"][joint_name]["y_global"],
                    self.current_ann["armature_keypoints"][joint_name]["z_global"],
                ]
                for joint_name in JOINT_NAMES[:55]
            ],
            dtype=np.float32,
        )
        translation_image_points = np.mean(image_points, axis=0)
        centered_image_points = image_points - translation_image_points
        smplx_points = joints[:55]
        translation_smplx_points = np.mean(smplx_points, axis=0)
        centered_smplx_points = smplx_points - translation_smplx_points

        R, loss = Rotation.align_vectors(centered_image_points, centered_smplx_points)
        # print("align vectors loss", loss)
        R = R.as_matrix()
        RT = np.concatenate([R, translation_image_points[:, np.newaxis]], axis=1)

        homogeneous_augmented_vertices = np.concatenate(
            [
                augmented_vertices - translation_smplx_points,
                np.ones([augmented_vertices.shape[0], 1]),
            ],
            axis=1,
        )

        return np.matmul(homogeneous_augmented_vertices, RT.T)

    def compute_2d_projection(
        self, joints: np.ndarray, vertices: np.ndarray, method: str = "pnp"
    ):
        if method == "pnp":
            self.compute_extrinsic_matrix(joints)
        elif method == "align_3d":
            vertices = self.align_3d_vertices(joints, vertices)

        vertices = np.concatenate([vertices, np.ones([vertices.shape[0], 1])], axis=1)
        vertices = np.matmul(vertices, self.current_extrinsic_matrix.T)
        vertices = np.matmul(vertices, self.K.T)
        vertices = vertices[:, :2] / vertices[:, 2:]

        self.current_vertices = vertices

        return vertices

    def compute_reprojection_accuracy(self):
        outside_vertices = 0
        for vertex in self.current_vertices:
            x, y = int(vertex[0]), int(vertex[1])
            if (
                0 <= x < self.current_seg.shape[0]
                and 0 <= y < self.current_seg.shape[1]
                and not self.current_seg[y, x].any()
            ):
                outside_vertices += 1

        accuracy = 1 - outside_vertices / self.current_vertices.shape[0]
        self.mean_accuracy.append(accuracy)

        return accuracy

    def get_mean_accuracy(self):
        return np.mean(self.mean_accuracy)
