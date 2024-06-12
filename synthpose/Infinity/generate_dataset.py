import glob
import json
import os
from copy import deepcopy

import cv2
import numpy as np
import reconstruction
import torch
from PIL import Image
from tqdm import tqdm
from video_scene import VideoScene

from ..constants import AUGMENTED_VERTICES_NAMES, COCO_VERTICES_NAME, MODEL_FOLDER


class DatasetGenerator:
    def __init__(
        self,
        data_folder: str,
        samples_per_video: int = 8,
        # sample_rate: int = 0.3,
        method="align_3d",
        output_path="infinity_dataset_combined",
        shift=0,
        split=0.8,
        infinity_version="v0.1.0",
    ):
        self.data_folder = data_folder
        self.samples_per_video = samples_per_video
        self.method = method
        self.output_path_train = os.path.join(output_path, "train")
        self.output_path_test = os.path.join(output_path, "test")
        self.shift = shift
        self.split = split
        if infinity_version == "v0.1.0":
            self.video_name = "video"
            self.labels_name = "labels"
            self.segmentation_name = "segmentation"
        else:
            self.video_name = "video.rgb"
            self.labels_name = "video.rgb"
            self.segmentation_name = "video.rgb"
        self.video_paths = sorted(
            glob.glob(os.path.join(data_folder, f"*/{self.video_name}.mp4"))
        )
        self.data_dict_train = {
            "infos": {},
            "images": [],
            "annotations": [],
            "categories": [],
        }

        self.data_dict_train["categories"] = [
            {
                "id": 0,
                "augmented_keypoints": AUGMENTED_VERTICES_NAMES,
                "coco_keypoints": COCO_VERTICES_NAME,
            }
        ]
        self.data_dict_test = deepcopy(self.data_dict_train)
        self.total_source_images = 0
        self.total_missing_bboxes = 0
        self.total_error_reconstruction = 0

        if not os.path.exists(os.path.join(self.output_path_train, "images")):
            os.makedirs(os.path.join(self.output_path_train, "images"))
        if not os.path.exists(os.path.join(self.output_path_test, "images")):
            os.makedirs(os.path.join(self.output_path_test, "images"))

    def save_image_annotation(
        self, annotation_dict, img, video_scene, index_frame, mode: str = "train"
    ):
        if mode == "train":
            self.data_dict_train["annotations"].append(annotation_dict)
            img_name = f"{len(self.data_dict_train['images'])}.png"
            img_path = os.path.join(self.output_path_train, "images", img_name)
            Image.fromarray(cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB)).save(
                img_path
            )
            image_dict = {
                "id": len(self.data_dict_train["images"]),
                "width": video_scene.dims[0],
                "height": video_scene.dims[1],
                "frame_number": index_frame,
                "img_path": img_path,
            }
            self.data_dict_train["images"].append(image_dict)
        elif mode == "test":
            self.data_dict_test["annotations"].append(annotation_dict)
            img_name = f"{len(self.data_dict_test['images'])}.png"
            img_path = os.path.join(self.output_path_test, "images", img_name)
            Image.fromarray(cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB)).save(
                img_path
            )
            image_dict = {
                "id": len(self.data_dict_test["images"]),
                "width": video_scene.dims[0],
                "height": video_scene.dims[1],
                "frame_number": index_frame,
                "img_path": img_path,
            }
            self.data_dict_test["images"].append(image_dict)

    def generate_dataset(self):
        train_size = int(len(self.video_paths) * self.split)
        mode = "train"
        for index_video, video_path in enumerate(tqdm(self.video_paths)):
            if index_video == train_size:
                mode = "test"
            path = os.path.dirname(video_path)
            video_scene = VideoScene(
                path_to_example=path,
                video_name=self.video_name,
                labels_name=self.labels_name,
                segmentation_name=self.segmentation_name,
            )
            indices_to_sample = list(
                set(
                    np.linspace(
                        0,
                        video_scene.nb_frames - 1,
                        self.samples_per_video,
                        dtype=np.int32,
                    )
                )
            )
            indices_to_sample = [index.item() for index in indices_to_sample]
            for index_frame in indices_to_sample:
                (
                    img,
                    ann,
                    groundtruth_landmarks,
                    coco_landmarks,
                    success,
                ) = self.get_grountruth_landmarks(video_scene, index_frame)

                self.total_source_images += 1
                if not success:
                    self.total_error_reconstruction += 1
                    continue
                if "bbox" not in ann:
                    self.total_missing_bboxes
                    continue

                annotation_dict = self.generate_annotation_dict(ann, mode=mode)
                annotation_dict["keypoints"] = groundtruth_landmarks
                annotation_dict["coco_keypoints"] = coco_landmarks
                self.save_image_annotation(
                    annotation_dict, img, video_scene, index_frame, mode=mode
                )

        with open(
            os.path.join(self.output_path_train, "annotations.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self.data_dict_train, f, ensure_ascii=False, indent=4)
        with open(
            os.path.join(self.output_path_test, "annotations.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self.data_dict_test, f, ensure_ascii=False, indent=4)

        print("total source images: ", self.total_source_images)
        print("total missing bboxes: ", self.total_missing_bboxes)
        print("total error reconstruction: ", self.total_error_reconstruction)

    def generate_annotation_dict(self, ann: dict, mode: str = "train"):
        annotation_dict = {}
        if mode == "train":
            annotation_dict["image_id"] = len(self.data_dict_train["images"])
        else:
            annotation_dict["image_id"] = len(self.data_dict_test["images"])
        annotation_dict["id"] = annotation_dict["image_id"]
        annotation_dict["category_id"] = 0
        annotation_dict["bbox"] = ann["bbox"]
        annotation_dict["percent_in_fov"] = ann["percent_in_fov"]
        annotation_dict["percent_occlusion"] = ann["percent_occlusion"]
        annotation_dict["iscrowd"] = 0

        return annotation_dict

    def get_grountruth_landmarks(self, video_scene: VideoScene, index_frame):
        img, ann, infos = video_scene.load_frame(index_frame)
        coco_landmarks = ann["keypoints"]
        gender = infos["avatar_presenting_gender"]
        betas = torch.tensor(infos["avatar_betas"], dtype=torch.float32).unsqueeze(0)
        poses = reconstruction.get_poses(ann)
        smplx_model = reconstruction.get_smplx_model(MODEL_FOLDER, gender, betas, poses)
        vertices, joints = reconstruction.get_vertices_and_joints(smplx_model, betas)
        augmented_vertices = reconstruction.get_augmented_vertices(vertices)
        projected_vertices = video_scene.compute_2d_projection(
            joints, augmented_vertices, method=self.method
        )
        if np.isnan(projected_vertices).any():
            return img, ann, {}, {}, False
        groundtruth_landmarks = {
            name: {"x": point[0], "y": point[1]}
            for name, point in zip(AUGMENTED_VERTICES_NAMES, projected_vertices)
        }

        # check if each landmark is out of frame (visible) or not:
        for name, point in groundtruth_landmarks.items():
            if (
                point["x"] < 0
                or point["y"] < 0
                or point["x"] > video_scene.dims[0]
                or point["y"] > video_scene.dims[1]
            ):
                groundtruth_landmarks[name]["v"] = 0
            else:
                groundtruth_landmarks[name]["v"] = 1

        return img, ann, groundtruth_landmarks, coco_landmarks, True


if __name__ == "__main__":
    dataset_generator = DatasetGenerator(
        data_folder="path_to_infinity_dataset",
        method="align_3d",
        output_path="infinity_reannotated",
        samples_per_video=15,
    )
    dataset_generator.generate_dataset()
