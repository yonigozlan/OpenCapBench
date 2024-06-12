import glob
import json
import os
import pickle

import numpy as np
import torch
from smplx import SMPLX
from tqdm import tqdm
from utils import CalibratedCamera

from ..constants import (
    AUGMENTED_VERTICES_INDEX_DICT,
    AUGMENTED_VERTICES_NAMES,
    BODIES_PATH,
    COCO_VERTICES_NAME,
    GENDERS_PATH,
    IMAGES_ROOT_PATH,
    IMGEXT,
    SCAN_CALIBRATION_PATH,
    SEQ_NAMES,
    SET,
    SMPLX_MODEL_DIR,
)


class DatasetGenerator:
    def __init__(
        self,
        output_path: str = "infinity_dataset_combined",
        sample_rate: int = 6,
    ):
        self.img_width = 4112
        self.img_height = 3008
        self.output_path = output_path
        self.sample_rate = sample_rate
        self.data_dict = {
            "infos": {},
            "images": [],
            "annotations": [],
            "categories": [],
        }

        self.data_dict["categories"] = [
            {
                "id": 0,
                "augmented_keypoints": AUGMENTED_VERTICES_NAMES,
                "coco_keypoints": COCO_VERTICES_NAME,
            }
        ]
        self.total_source_images = 0
        self.total_error_reconstruction = 0

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def get_bbox(self, vertices):
        x_img, y_img = vertices[:, 0], vertices[:, 1]
        xmin = min(x_img)
        ymin = min(y_img)
        xmax = max(x_img)
        ymax = max(y_img)

        x_center = (xmin + xmax) / 2.0
        width = xmax - xmin
        xmin = x_center - 0.5 * width  # * 1.2
        xmax = x_center + 0.5 * width  # * 1.2

        y_center = (ymin + ymax) / 2.0
        height = ymax - ymin
        ymin = y_center - 0.5 * height  # * 1.2
        ymax = y_center + 0.5 * height  # * 1.2

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(self.img_width, xmax)
        ymax = min(self.img_height, ymax)

        bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(int)

        return bbox

    def generate_annotation_dict(self):
        annotation_dict = {}
        annotation_dict["image_id"] = len(self.data_dict["images"])
        annotation_dict["id"] = annotation_dict["image_id"]
        annotation_dict["category_id"] = 0
        annotation_dict["iscrowd"] = 0

        return annotation_dict

    def get_grountruth_landmarks(
        self,
        body_model,
        set: str,
        cam,
        seq_name: str,
        frame_id: int,
        sub_id,
    ):
        smplx_params_fn = os.path.join(
            BODIES_PATH, set, seq_name, f"{frame_id:05d}", f"{sub_id}.pkl"
        )
        body_params = pickle.load(open(smplx_params_fn, "rb"))
        body_params = {k: torch.from_numpy(v) for k, v in body_params.items()}
        body_model.reset_params(**body_params)
        model_output = body_model(
            return_verts=True, body_pose=body_params["body_pose"], return_full_pose=True
        )
        vertices = model_output.vertices.detach()

        ## project to image

        projected_vertices = cam(vertices).squeeze().detach().numpy()
        bbox = self.get_bbox(projected_vertices)
        projected_vertices_anatomical = projected_vertices[
            list(AUGMENTED_VERTICES_INDEX_DICT.values())
        ].tolist()

        coco_landmarks = [0] * 3 * 17

        if np.isnan(projected_vertices_anatomical).any():
            return {}, {}, False
        groundtruth_landmarks = {
            name: {"x": point[0], "y": point[1]}
            for name, point in zip(
                AUGMENTED_VERTICES_NAMES, projected_vertices_anatomical
            )
        }

        # check if each landmark is out of frame (visible) or not:
        for name, point in groundtruth_landmarks.items():
            if (
                point["x"] < 0
                or point["y"] < 0
                or point["x"] > self.img_width
                or point["y"] > self.img_height
            ):
                groundtruth_landmarks[name]["v"] = 0
            else:
                groundtruth_landmarks[name]["v"] = 1

        return groundtruth_landmarks, coco_landmarks, bbox, True

    def generate_dataset(self):
        iteration = 0
        it_file = 0
        nb_files = len(SEQ_NAMES)
        gender_mapping = json.load(open(GENDERS_PATH, "r"))

        for seq_name in SEQ_NAMES:
            print("extracting data from ", seq_name, "...")
            splits = seq_name.split("_")
            if len(splits) == 3:
                scene_name, sub_id, _ = splits
                sub_ids = [sub_id]
            else:
                scene_name = splits[0]
                sub_ids = splits[1:-1]
            # extension = imgext[scene_name]
            extension = IMGEXT[scene_name]
            for sub_id in sub_ids:
                print("sub_id: ", sub_id)
                gender = gender_mapping[f"{int(sub_id)}"]

                seq_path = os.path.join(IMAGES_ROOT_PATH, SET, seq_name)
                cams_paths = [
                    item
                    for item in os.listdir(seq_path)
                    if os.path.isdir(os.path.join(seq_path, item))
                ]
                body_model = SMPLX(
                    SMPLX_MODEL_DIR,
                    gender=gender,
                    num_pca_comps=12,
                    flat_hand_mean=False,
                    create_expression=True,
                    create_jaw_pose=True,
                )
                cam_iter = 0
                nb_cams = len(cams_paths)
                for cam_path in cams_paths:
                    camera_id = int(cam_path.split("_")[-1])
                    print("camera_id: ", camera_id)
                    calib_path = os.path.join(
                        SCAN_CALIBRATION_PATH,
                        scene_name,
                        "calibration",
                        f"{camera_id:03d}.xml",
                    )
                    if not os.path.exists(calib_path):
                        print(
                            f"skipped camera {camera_id} for scene {scene_name} because no calibration file"
                        )
                        continue

                    cam = CalibratedCamera(calib_path=calib_path)
                    images_paths = glob.glob(
                        os.path.join(seq_path, cam_path) + f"/*.{extension}"
                    )
                    nb_images = len(images_paths) // self.sample_rate
                    for index_frame, image_path in enumerate(images_paths):
                        frame_id = int(image_path.split("/")[-1].split("_")[0])

                        if not os.path.exists(
                            os.path.join(
                                BODIES_PATH,
                                SET,
                                seq_name,
                                f"{frame_id:05d}",
                                f"{sub_id}.pkl",
                            )
                        ):
                            continue
                        if index_frame % self.sample_rate == 0:
                            (
                                groundtruth_landmarks,
                                coco_landmarks,
                                bbox,
                                success,
                            ) = self.get_grountruth_landmarks(
                                body_model,
                                SET,
                                cam,
                                seq_name,
                                frame_id,
                                sub_id,
                            )

                            self.total_source_images += 1
                            if not success:
                                self.total_error_reconstruction += 1
                                continue

                            annotation_dict = self.generate_annotation_dict()
                            annotation_dict["bbox"] = bbox.tolist()
                            annotation_dict["keypoints"] = groundtruth_landmarks
                            annotation_dict["coco_keypoints"] = coco_landmarks

                            self.data_dict["annotations"].append(annotation_dict)

                            image_dict = {
                                "id": len(self.data_dict["images"]),
                                "width": self.img_width,
                                "height": self.img_height,
                                "frame_number": index_frame,
                                "img_path": image_path,
                            }
                            self.data_dict["images"].append(image_dict)

                            if iteration % 100 == 0:
                                print(
                                    f"scene {it_file}/{nb_files}, cam {cam_iter}/{nb_cams}, {index_frame//self.sample_rate}/{nb_images}"
                                )
                            iteration += 1
                    cam_iter += 1
            it_file += 1

        with open(
            os.path.join(self.output_path, "annotations.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self.data_dict, f, ensure_ascii=False, indent=4)

        print("total source images: ", self.total_source_images)
        print("total error reconstruction: ", self.total_error_reconstruction)


if __name__ == "__main__":
    dataset_generator = DatasetGenerator(
        output_path="rich_reannotated",
        sample_rate=6,
    )
    dataset_generator.generate_dataset()
