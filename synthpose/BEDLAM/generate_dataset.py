import glob
import json
import os

import numpy as np
import smplx
import torch

from ..constants import (
    AUGMENTED_VERTICES_INDEX_DICT,
    AUGMENTED_VERTICES_NAMES,
    COCO_VERTICES_NAME,
    MODEL_FOLDER,
)

ROTATION_MATRIX_2D_90_CLOCKWISE = np.array([[0, 1], [-1, 0]])

smplx_model_male = smplx.create(
    MODEL_FOLDER,
    model_type="smplx",
    gender="neutral",
    ext="npz",
    flat_hand_mean=True,
    num_betas=11,
    use_pca=False,
)
smplx_model_female = smplx.create(
    MODEL_FOLDER,
    model_type="smplx",
    gender="female",
    ext="npz",
    num_betas=11,
    flat_hand_mean=True,
    use_pca=False,
)

smplx_model_neutral = smplx.create(
    MODEL_FOLDER,
    model_type="smplx",
    gender="neutral",
    ext="npz",
    flat_hand_mean=True,
    num_betas=11,
    use_pca=False,
)


def get_smplx_vertices(poses, betas, trans, gender):
    if gender == "male":
        model_out = smplx_model_male(
            betas=torch.tensor(betas).unsqueeze(0).float(),
            global_orient=torch.tensor(poses[:3]).unsqueeze(0).float(),
            body_pose=torch.tensor(poses[3:66]).unsqueeze(0).float(),
            left_hand_pose=torch.tensor(poses[75:120]).unsqueeze(0).float(),
            right_hand_pose=torch.tensor(poses[120:165]).unsqueeze(0).float(),
            jaw_pose=torch.tensor(poses[66:69]).unsqueeze(0).float(),
            leye_pose=torch.tensor(poses[69:72]).unsqueeze(0).float(),
            reye_pose=torch.tensor(poses[72:75]).unsqueeze(0).float(),
            transl=torch.tensor(trans).unsqueeze(0),
        )
    # from psbody.mesh import Mesh
    elif gender == "female":
        model_out = smplx_model_female(
            betas=torch.tensor(betas).unsqueeze(0).float(),
            global_orient=torch.tensor(poses[:3]).unsqueeze(0).float(),
            body_pose=torch.tensor(poses[3:66]).unsqueeze(0).float(),
            left_hand_pose=torch.tensor(poses[75:120]).unsqueeze(0).float(),
            right_hand_pose=torch.tensor(poses[120:165]).unsqueeze(0).float(),
            jaw_pose=torch.tensor(poses[66:69]).unsqueeze(0).float(),
            leye_pose=torch.tensor(poses[69:72]).unsqueeze(0).float(),
            reye_pose=torch.tensor(poses[72:75]).unsqueeze(0).float(),
            transl=torch.tensor(trans).unsqueeze(0),
        )
    elif gender == "neutral":
        model_out = smplx_model_neutral(
            betas=torch.tensor(betas).unsqueeze(0).float(),
            global_orient=torch.tensor(poses[:3]).unsqueeze(0).float(),
            body_pose=torch.tensor(poses[3:66]).unsqueeze(0).float(),
            left_hand_pose=torch.tensor(poses[75:120]).unsqueeze(0).float(),
            right_hand_pose=torch.tensor(poses[120:165]).unsqueeze(0).float(),
            jaw_pose=torch.tensor(poses[66:69]).unsqueeze(0).float(),
            leye_pose=torch.tensor(poses[69:72]).unsqueeze(0).float(),
            reye_pose=torch.tensor(poses[72:75]).unsqueeze(0).float(),
            transl=torch.tensor(trans).unsqueeze(0),
        )
    else:
        print("Please provide gender as male or female")
    return model_out.vertices[0], model_out.joints[0]


class DatasetGenerator:
    def __init__(
        self,
        annotation_files_path: str,
        output_path: str = "infinity_dataset_combined",
        sample_rate: int = 6,
    ):
        self.img_width = 1280
        self.img_height = 720
        self.annotation_files_path = annotation_files_path
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
        self, annotations: dict, index_frame: int, rotate_flag: bool = False
    ):
        pose = annotations["pose_cam"][index_frame]
        beta = annotations["shape"][index_frame]
        body_trans_cam = annotations["trans_cam"][index_frame]
        verts_3d, joints_3d = get_smplx_vertices(pose, beta, body_trans_cam, "neutral")
        cam_trans = annotations["cam_ext"][index_frame][:, 3][:3]
        verts_3d = verts_3d.detach().cpu() + cam_trans

        projected_vertices = np.matmul(
            annotations["cam_int"][index_frame], verts_3d.T
        ).T
        projected_vertices = projected_vertices[:, :2] / projected_vertices[:, 2:]
        projected_vertices = projected_vertices.numpy()

        if rotate_flag:
            projected_vertices[:] -= np.array([self.img_height / 2, self.img_width / 2])
            projected_vertices = np.matmul(
                ROTATION_MATRIX_2D_90_CLOCKWISE, projected_vertices.T
            ).T
            projected_vertices[:] += np.array([self.img_width / 2, self.img_height / 2])

        projected_vertices_anatomical = projected_vertices[
            list(AUGMENTED_VERTICES_INDEX_DICT.values())
        ]

        bbox = self.get_bbox(projected_vertices)

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
        annotations_files_paths = glob.glob(self.annotation_files_path + "/*.npz")
        iteration = 0
        it_file = 0
        nb_files = len(annotations_files_paths)
        print("creating dataset annotations...")
        print("using cuda: ", torch.cuda.is_available())
        for annotation_path in annotations_files_paths:
            rotate_flag = False
            if "closeup" in annotation_path:  # Since the original image are rotated
                rotate_flag = True
            annotations = np.load(annotation_path)
            nb_images = len(annotations["imgname"]) // self.sample_rate
            print("starting file: ", annotation_path)
            for index_frame, img_name in enumerate(annotations["imgname"]):
                if index_frame % self.sample_rate == 0:
                    (
                        groundtruth_landmarks,
                        coco_landmarks,
                        bbox,
                        success,
                    ) = self.get_grountruth_landmarks(
                        annotations, index_frame, rotate_flag
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
                        "img_path": os.path.join(
                            annotation_path.split("/")[-1].split(".")[0],
                            "png",
                            img_name,
                        ),
                    }
                    self.data_dict["images"].append(image_dict)

                    if iteration % 100 == 0:
                        print(
                            f"scene {it_file}/{nb_files}, {index_frame//self.sample_rate}/{nb_images}"
                        )
                    iteration += 1

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
        annotation_files_path="../all_npz_12_validation",
        output_path="bedlam_reannotated",
        sample_rate=6,
    )
    dataset_generator.generate_dataset()
