import pickle
from typing import Optional, Sequence, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from constants import (
    getMMposeAnatomicalCocoMarkerNames,
    getMMposeAnatomicalCocoMarkerPairs,
    getMMposeMarkerNames,
)
from mmcv.ops import RoIPool
from mmcv.transforms import Compose
from mmdet.apis import init_detector
from mmdet.structures import DetDataSample, SampleList
from mmdet.utils import get_test_pipeline_cfg
from mmengine.dataset import Compose, pseudo_collate
from mmpose.apis import init_model as init_pose_estimator
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def detection_inference(
    model_config,
    model_ckpt,
    video_path,
    bbox_path,
    batch_size=32,
    device="cuda:0",
    det_cat_id=1,
):
    """Visualize the demo images.

    Using mmdet to detect the human.
    """

    det_model = init_detector(model_config, model_ckpt, device=device.lower())

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Faild to load video file {video_path}"

    output = []
    nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    batched_frames = []
    for img in tqdm(frame_iter(cap), total=nFrames):
        # test a batch of images
        batched_frames.append(img)
        if len(batched_frames) == batch_size:
            # the resulting box is (x1, y1, x2, y2)
            mmdet_results_batched = inference_detector_batched(
                det_model, batched_frames
            )
            batched_frames = []
            for mmdet_results in mmdet_results_batched:
                # keep the person class bounding boxes.
                person_results = process_mmdet_results(mmdet_results, det_cat_id)
                output.append(person_results)

    if len(batched_frames) > 0:
        mmdet_results_batched = inference_detector_batched(det_model, batched_frames)
        batched_frames = []
        for mmdet_results in mmdet_results_batched:
            # keep the person class bounding boxes.
            person_results = process_mmdet_results(mmdet_results, det_cat_id)
            output.append(person_results)

    output_file = bbox_path
    pickle.dump(output, open(str(output_file), "wb"))
    cap.release()


def inference_detector_batched(
    model: nn.Module,
    imgs: Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]],
    test_pipeline: Optional[Compose] = None,
    text_prompt: Optional[str] = None,
    custom_entities: bool = False,
) -> Union[DetDataSample, SampleList]:
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str, ndarray, Sequence[str/ndarray]):
           Either image files or loaded images.
        test_pipeline (:obj:`Compose`): Test pipeline.

    Returns:
        :obj:`DetDataSample` or list[:obj:`DetDataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg

    if test_pipeline is None:
        cfg = cfg.copy()
        test_pipeline = get_test_pipeline_cfg(cfg)
        if isinstance(imgs[0], np.ndarray):
            # Calling this method across libraries will result
            # in module unregistered error if not prefixed with mmdet.
            test_pipeline[0].type = "mmdet.LoadImageFromNDArray"

        test_pipeline = Compose(test_pipeline)

    if model.data_preprocessor.device.type == "cpu":
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), "CPU inference with RoIPool is not supported currently."

    result_list = []
    data_batched = {}
    data_batched["inputs"] = []
    data_batched["data_samples"] = []
    for i, img in enumerate(imgs):
        # prepare data
        if isinstance(img, np.ndarray):
            # TODO: remove img_id.
            data_ = dict(img=img, img_id=0)
        else:
            # TODO: remove img_id.
            data_ = dict(img_path=img, img_id=0)

        if text_prompt:
            data_["text"] = text_prompt
            data_["custom_entities"] = custom_entities

        # build the data pipeline
        data_ = test_pipeline(data_)

        data_batched["inputs"].append(data_["inputs"])
        data_batched["data_samples"].append(data_["data_samples"])

        # forward the model
    with torch.no_grad():
        result_list = model.test_step(data_batched)

    torch.cuda.empty_cache()

    if not is_batch:
        return result_list[0]
    else:
        return result_list


def pose_inference_updated(
    model_config,
    model_ckpt,
    video_path,
    bbox_path,
    pkl_path,
    video_out_path,
    device="cuda:0",
    batch_size=8,
    bbox_thr=0.95,
    visualize=True,
    save_results=True,
    marker_set="Anatomical",
):
    """Run pose inference on custom video dataset"""

    # init model
    model = init_pose_estimator(model_config, model_ckpt, device)
    model_name = model_config.split("/")[1].split(".")[0]
    print("Initializing {} Model".format(model_name))

    # build data pipeline
    # test_pipeline = init_test_pipeline(model)
    test_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    # build dataset
    video_basename = video_path.split("/")[-1].split(".")[0]
    dataset = CustomVideoDataset(
        video_path=video_path,
        bbox_path=bbox_path,
        bbox_threshold=bbox_thr,
        pipeline=test_pipeline,
        config=model.cfg,
        dataset_meta=model.dataset_meta,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=pseudo_collate
    )
    print("Building {} Custom Video Dataset".format(video_basename))

    # run pose inference
    print("Running pose inference...")
    instances = []
    for batch in tqdm(dataloader):
        # print("batch data_samples", batch["data_samples"][0].keys())
        # print("batch inputs", batch["inputs"].keys())
        # batch["img"] = batch["img"].to(device)
        # batch["img_metas"] = [img_metas[0] for img_metas in batch["img_metas"].data]
        with torch.no_grad():
            # result = run_pose_inference(model, batch)
            results = model.test_step(batch)
        instances += results

    # concat results and transform to per frame format

    # results = concat(instances)
    results = merge_data_samples(instances)
    results = convert_instance_to_frame(results, dataset.frame_to_instance)
    # print("results", results, len(results), results[0][0].keys())
    # run pose tracking
    # results = run_pose_tracking(results)

    # save results
    if save_results:
        print("Saving Pose Results...")
        kpt_save_file = pkl_path
        with open(kpt_save_file, "wb") as f:
            pickle.dump(results, f)

    # visualzize
    if visualize:
        model.cfg.visualizer.radius = 3
        model.cfg.visualizer.alpha = 0.8
        model.cfg.visualizer.line_width = 1
        print("Rendering Visualization...")
        visualizer = VISUALIZERS.build(model.cfg.visualizer)
        visualizer.set_dataset_meta(model.dataset_meta, skeleton_style="mmpose")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_save_file = video_out_path
        videoWriter = cv2.VideoWriter(str(video_save_file), fourcc, fps, size)
        # markerPairs = getMMposeAnatomicalMarkerPairs()
        # markerNames = getMMposeAnatomicalMarkerNames()
        if marker_set == "Anatomical":
            markerPairs = getMMposeAnatomicalCocoMarkerPairs()
            markerNames = getMMposeAnatomicalCocoMarkerNames()
            for pose_results, img in tqdm(zip(results, frame_iter(cap))):
                # display keypoints and bbox on frame
                preds = pose_results[0]["pred_instances"]
                for index_person in range(len(preds["bboxes"])):
                    bbox = preds["bboxes"][index_person]
                    kpts = preds["keypoints"][index_person]
                    # cv2.rectangle(
                    #     img,
                    #     (int(bbox[0]), int(bbox[1])),
                    #     (int(bbox[2]), int(bbox[3])),
                    #     (0, 255, 0),
                    #     2,
                    # )
                    for index_kpt in range(len(kpts)):
                        if index_kpt < 17:
                            color = (0, 0, 255)

                        else:
                            # if markerNames[index_kpt-17] in markerPairs.keys():
                            #     if markerNames[index_kpt-17][0] == "l":
                            if markerNames[index_kpt] in markerPairs.keys():
                                if markerNames[index_kpt][0] == "l":
                                    color = (255, 255, 0)
                                else:
                                    color = (255, 0, 255)
                            else:
                                color = (255, 0, 0)

                        cv2.circle(
                            img,
                            (int(kpts[index_kpt][0]), int(kpts[index_kpt][1])),
                            4,
                            color,
                            -1,
                        )
                videoWriter.write(img)
        elif marker_set == "Coco":
            for pose_results, img in tqdm(zip(results, frame_iter(cap))):
                # display keypoints and bbox on frame
                preds = pose_results[0]["pred_instances"]
                for index_person in range(len(preds["bboxes"])):
                    bbox = preds["bboxes"][index_person]
                    kpts = preds["keypoints"][index_person]
                    # cv2.rectangle(
                    #     img,
                    #     (int(bbox[0]), int(bbox[1])),
                    #     (int(bbox[2]), int(bbox[3])),
                    #     (0, 255, 0),
                    #     2,
                    # )
                    for index_kpt in range(len(getMMposeMarkerNames())):
                        color = (0, 0, 255)
                        cv2.circle(
                            img,
                            (int(kpts[index_kpt][0]), int(kpts[index_kpt][1])),
                            4,
                            color,
                            -1,
                        )
                videoWriter.write(img)
        videoWriter.release()


def frame_iter(capture):
    while capture.grab():
        yield capture.retrieve()[1]


def convert_instance_to_frame(results, frame_to_instance):
    """Convert pose results from per instance to per frame format
    Args:
        results dict(array): dict of saved outputs
        frame_to_instance list: list of instance idx per frame
    Returns:
        results list(list(dict)): frame list of every instance's result dict
    """
    results_frame = []
    for idxs in frame_to_instance:
        results_frame.append([])
        for idx in idxs:
            result_instance = {k: v[idx] for k, v in results.items()}
            results_frame[-1].append(result_instance)

    return results_frame


def process_mmdet_results(mmdet_results, cat_id=1):
    """Process mmdet results, and return a list of bboxes.

    :param mmdet_results:
    :param cat_id: category id (default: 1 for human)
    :return: a list of detected bounding boxes
    """
    if isinstance(mmdet_results, tuple):
        det_results = mmdet_results[0]
    else:
        det_results = mmdet_results
    pred_instances = det_results.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instances.bboxes, pred_instances.scores[:, None]), axis=1
    )
    bboxes = bboxes[pred_instances.labels == cat_id - 1, :]
    # bboxes = det_results[cat_id - 1]
    person_results = []
    for bbox in bboxes:
        person = {}
        person["bbox"] = bbox
        person_results.append(person)

    return person_results


class CustomVideoDataset(Dataset):
    """Create custom video dataset for top down inference

    Args:
        video_path (str): Path to video file
        bbox_path (str): Path to bounding box file
                         (expects format to be xyxy [left, top, right, bottom])
        pipeline (list[dict | callable]): A sequence of data transforms
    """

    def __init__(
        self, video_path, bbox_path, bbox_threshold, pipeline, config, dataset_meta
    ):
        # load video
        self.capture = cv2.VideoCapture(video_path)
        assert self.capture.isOpened(), f"Failed to load video file {video_path}"
        self.frames = np.stack([x for x in frame_iter(self.capture)])

        # load bbox
        self.bboxs = pickle.load(open(bbox_path, "rb"))
        print(f"Loaded {len(self.bboxs)} frames from {video_path}")
        self.bbox_threshold = bbox_threshold

        # create instance to frame and frame to instance mapping
        self.instance_to_frame = []
        self.frame_to_instance = []
        for i, frame in enumerate(self.bboxs):
            self.frame_to_instance.append([])
            for j, instance in enumerate(frame):
                bbox = instance["bbox"]
                if bbox[4] >= bbox_threshold:
                    self.instance_to_frame.append([i, j])
                    self.frame_to_instance[-1].append(len(self.instance_to_frame) - 1)

        self.pipeline = pipeline
        self.cfg = config
        self.dataset_meta = dataset_meta
        # flip_pair_dict = get_flip_pair_dict()
        # self.flip_pairs = flip_pair_dict[self.cfg.data.test.type]
        self.flip_pairs = dataset_meta.get("flip_pairs", None)

    def _xyxy2xywh(self, bbox_xyxy):
        """Transform the bbox format from x1y1x2y2 to xywh.
        Args:
            bbox_xyxy (np.ndarray): Bounding boxes (with scores), shaped (4,) or
                (5,). (left, top, right, bottom, [score])
        Returns:
            np.ndarray: Bounding boxes (with scores),
            shaped (4,) or (5,). (left, top, width, height, [score])
        """
        bbox_xywh = bbox_xyxy.copy()
        bbox_xywh[2] = bbox_xywh[2] - bbox_xywh[0] + 1
        bbox_xywh[3] = bbox_xywh[3] - bbox_xywh[1] + 1
        return bbox_xywh

    def _box2cs(self, cfg, box):
        """This encodes bbox(x,y,w,h) into (center, scale)
        Args:
            x, y, w, h
        Returns:
            tuple: A tuple containing center and scale.
            - np.ndarray[float32](2,): Center of the bbox (x, y).
            - np.ndarray[float32](2,): Scale of the bbox w & h.
        """
        x, y, w, h = box[:4]
        # input_size = cfg.data_cfg["image_size"]
        input_size = cfg.codec["input_size"]
        aspect_ratio = input_size[0] / input_size[1]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        # pixel std is 200.0
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)

        scale = scale * 1.25
        return center, scale

    def __len__(self):
        return len(self.instance_to_frame)

    def __getitem__(self, idx):
        frame_num, detection_num = self.instance_to_frame[idx]
        # num_joints = self.cfg.data_cfg["num_joints"]
        num_joints = self.dataset_meta["num_keypoints"]
        bbox_xyxy = self.bboxs[frame_num][detection_num]["bbox"]
        bbox_xywh = self._xyxy2xywh(bbox_xyxy)
        center, scale = self._box2cs(self.cfg, bbox_xywh)

        # joints_3d and joints_3d_visalble are place holders
        # but bbox in image file, image file is not used but we need bbox information later
        data = {
            "img": self.frames[frame_num],
            "bbox": bbox_xyxy[None, :4],
            # "bbox_center": center[None, :],
            # "bbox_scale": scale[None, :],
            # "bbox_score": bbox_xywh[4] if len(bbox_xywh) == 5 else 1,
            "bbox_score": np.ones(1, dtype=np.float32),
            "bbox_id": 0,
            "joints_3d": np.zeros((num_joints, 3)),
            "joints_3d_visible": np.zeros((num_joints, 3)),
            "rotation": 0,
            "ann_info": {
                "image_size": np.array(self.cfg.codec["input_size"]),
                "num_joints": num_joints,
                "flip_pairs": self.flip_pairs,
            },
        }
        # data["bbox_score"] = data["bbox_score"][None]
        data.update(self.dataset_meta)
        data = self.pipeline(data)
        return data
