import json
import os

from PIL import Image
from tqdm import tqdm


def downsample_images(
    source_folder, annotation_file, write_images=False, write_annotations=True
):
    # Create a new folder in the source_folder to store the downsampled images and annotations
    dest_folder = os.path.join(source_folder, "downsampled")
    os.makedirs(dest_folder, exist_ok=True)
    orig_width = 4112
    orig_height = 3008
    new_width = 984
    new_height = 720
    ratio = new_width / orig_width

    if write_images:
        # Traverse through the source folder and find all the image files
        cnt_images = 0
        for root, dirs, files in os.walk(
            os.path.join(source_folder, "ps/project/multi-ioi/rich_release")
        ):
            for file in files:
                # if file.endswith('.bmp') or file.endswith('.png'):
                if file.endswith(".png"):
                    # Open the image file and downsample it to 1280x720
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path)
                    img = img.resize((new_width, new_height), Image.ANTIALIAS)

                    # Save the downsampled image in the new folder with the same nested architecture
                    dest_path = os.path.join(
                        dest_folder, os.path.relpath(img_path, source_folder)
                    )
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    dest_path = dest_path[:-3] + "jpg"
                    img.save(dest_path, "JPEG")
                    cnt_images += 1
                    if cnt_images % 100 == 0:
                        print(f"Saved {cnt_images} images")
        print(f"Saved {cnt_images} images in total")
    if write_annotations:
        # Downsample the corresponding annotations
        with open(annotation_file) as f:
            annotations = json.load(f)
        for annotation in annotations["images"]:
            annotation["width"] = new_width
            annotation["height"] = new_height
            new_img_path = os.path.join(
                dest_folder,
                os.path.relpath(annotation["img_path"], source_folder).removeprefix(
                    "../../"
                ),
            )
            new_img_path = new_img_path[:-3] + "jpg"
            annotation["img_path"] = new_img_path
        for annotation in annotations["annotations"]:
            annotation["bbox"] = [round(x * ratio) for x in annotation["bbox"]]
            for keypoint_name, keypoint in annotation["keypoints"].items():
                keypoint["x"] = round(keypoint["x"] * ratio)
                keypoint["y"] = round(keypoint["y"] * ratio)
            for index, coord in enumerate(annotation["coco_keypoints"]):
                if index % 3 != 2:
                    annotation["coco_keypoints"][index] = round(coord * ratio)
        # Save the modified annotations in a new JSON file with the same nested architecture
        dest_annotation_file = os.path.join(
            dest_folder, os.path.relpath(annotation_file, source_folder)
        )
        os.makedirs(os.path.dirname(dest_annotation_file), exist_ok=True)
        with open(dest_annotation_file, "w") as f:
            json.dump(annotations, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    downsample_images(
        "path_to_rich_dataset",
        "rich_downsampled/annotations.json",
        write_images=False,
        write_annotations=True,
    )
