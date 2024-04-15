import os
import pickle
import shutil

import numpy as np
import scipy
from constants import (
    getMMposeAnatomicalCocoMarkerNames,
    getMMposeMarkerNames,
    getOpenPoseMarkerNames,
)


def runPoseDetector(
    config_benchmark,
    CameraDirectories,
    trialRelativePath,
    pathPoseDetector,
    trialName,
    CamParamDict=None,
    generateVideo=True,
    cams2Use=["all"],
    poseDetector="OpenPose",
    bbox_thr=0.8,
):
    # Create list of cameras.
    if cams2Use[0] == "all":
        cameras2Use = list(CameraDirectories.keys())
    else:
        cameras2Use = cams2Use

    CameraDirectories_selectedCams = {}
    CamParamList_selectedCams = []
    for cam in cameras2Use:
        CameraDirectories_selectedCams[cam] = CameraDirectories[cam]
        CamParamList_selectedCams.append(CamParamDict[cam])

    # Get/add video extension.
    cameraDirectory = CameraDirectories_selectedCams[cameras2Use[0]]
    trialRelativePath += ".avi"

    for camName in CameraDirectories_selectedCams:
        cameraDirectory = CameraDirectories_selectedCams[camName]
        print("Running {} for {}".format(poseDetector, camName))
        runPoseVideo(
            config_benchmark,
            cameraDirectory,
            trialRelativePath,
            pathPoseDetector,
            trialName,
            generateVideo=generateVideo,
            bbox_thr=bbox_thr,
        )

    return


def runPoseVideo(
    config_benchmark,
    cameraDirectory,
    fileName,
    pathMMpose,
    trialName,
    generateVideo=True,
    bbox_thr=0.8,
):
    model_config_person = config_benchmark["model_config_person"]
    model_ckpt_person = config_benchmark["model_ckpt_person"]
    model_config_pose = config_benchmark["model_config_pose"]
    trialPrefix, _ = os.path.splitext(os.path.basename(fileName))
    videoFullPath = os.path.normpath(os.path.join(cameraDirectory, fileName))

    pathOutputVideo = os.path.join(
        cameraDirectory, "OutputMedia_mmpose_" + str(bbox_thr), trialName
    )

    # mmposeBoxDir = os.path.join("OutputBox_mmpose", trialName)
    pathOutputBox = config_benchmark["OutputBoxDirectory"]
    pathOutputBox = os.path.join(
        config_benchmark["OutputBoxDirectory"].join(
            [
                config_benchmark["dataName"].join(cameraDirectory.split(config_benchmark["dataName"])[:-1]),
                cameraDirectory.split(config_benchmark["dataName"])[-1]
            ]
        ),
        trialName,
    )
    # pathOutputBox = os.path.join(cameraDirectory, mmposeBoxDir)

    mmposePklDir = os.path.join("OutputPkl_mmpose_" + str(bbox_thr), trialName)
    pathOutputPkl = os.path.join(cameraDirectory, mmposePklDir)

    os.makedirs(pathOutputVideo, exist_ok=True)
    os.makedirs(pathOutputPkl, exist_ok=True)

    # The video is rewritten, unrotated, and downsampled. There is no
    # need to do anything specific for the rotation, just rewriting the video
    # unrotates it.
    trialPath, _ = os.path.splitext(fileName)
    fileName = trialPath + "_rotated.avi"
    pathVideoRot = os.path.normpath(os.path.join(cameraDirectory, fileName))
    cmd_fr = " "
    # if frameRate > 60.0:
    #     cmd_fr = ' -r 60 '
    #     frameRate = 60.0
    CMD = "ffmpeg -loglevel error -y -i {}{}-q 0 {}".format(
        videoFullPath, cmd_fr, pathVideoRot
    )

    videoFullPath = pathVideoRot
    trialPrefix = trialPrefix + "_rotated"

    if not os.path.exists(pathVideoRot):
        os.system(CMD)

    pklPath = os.path.join(pathOutputPkl, trialPrefix + ".pkl")
    ppPklPath = os.path.join(pathOutputPkl, trialPrefix + "_pp.pkl")
    # Run pose detector if this file doesn't exist in outputs
    if not os.path.exists(ppPklPath):
        if config_benchmark["alt_model"] == "VirtualMarker":
            from utilsMMpose import detection_inference
            from utilsPose import pose_inference_updated
        else:
            from utilsMMpose import detection_inference, pose_inference_updated

        # Run human detection.
        bboxPath = os.path.join(pathOutputBox, trialPrefix + ".pkl")
        print("bboxPath", bboxPath)
        if not os.path.exists(bboxPath):
            os.makedirs(pathOutputBox, exist_ok=True)
            full_model_config_person = os.path.join(pathMMpose, model_config_person)
            detection_inference(
                full_model_config_person,
                model_ckpt_person,
                videoFullPath,
                bboxPath,
                batch_size=config_benchmark["batch_size_det"],
            )

        # Run pose detection.
        pathModelCkptPose = config_benchmark["model_ckpt_pose_absolute"]
        videoOutPath = os.path.join(pathOutputVideo, trialPrefix + "withKeypoints.mp4")
        full_model_config_pose = os.path.join(pathMMpose, model_config_pose)
        pose_inference_updated(
            full_model_config_pose,
            pathModelCkptPose,
            videoFullPath,
            bboxPath,
            pklPath,
            videoOutPath,
            batch_size=config_benchmark["batch_size_pose"],
            bbox_thr=bbox_thr,
            visualize=generateVideo,
            marker_set=config_benchmark["marker_set"],
        )

    # Post-process data to have OpenPose-like file structure.
    # arrangeMMposePkl(pklPath, ppPklPath)
    if config_benchmark["alt_model"] is None:
        if config_benchmark["marker_set"] == "Anatomical":
            arrangeMMposeAnatomicalPkl(pklPath, ppPklPath)
        elif config_benchmark["marker_set"] == "Coco":
            arrangeMMposePkl(pklPath, ppPklPath)
    else:
        # copy pklPath to ppPklPath
        shutil.copy(pklPath, ppPklPath)


def arrangeMMposeAnatomicalPkl(poseInferencePklPath, outputPklPath):
    open_file = open(poseInferencePklPath, "rb")
    frames = pickle.load(open_file)
    open_file.close()

    markersMMposeAnatomical = getMMposeAnatomicalCocoMarkerNames()
    # markersMMposeAnatomical = getMMposeAnatomicalMarkerNames()
    nb_markers = len(markersMMposeAnatomical)
    data4pkl = []
    for c_frame, frame in enumerate(frames):
        data4people = []
        for c, person in enumerate(frame):
            coordinates = person["pred_instances"]["keypoints"][0, :, :]
            c_coord_out = np.zeros((nb_markers * 3,))
            for c_m, marker in enumerate(markersMMposeAnatomical):
                c_coord = [coordinates[c_m][0], coordinates[c_m][1]]
                c_coord.append(person["pred_instances"]["keypoint_scores"][0][c_m])
                idx_out = np.arange(c_m * 3, c_m * 3 + 3)
                c_coord_out[idx_out,] = c_coord
            c_dict = {}
            c_dict["person_id"] = [c]
            c_dict["pose_keypoints_2d"] = c_coord_out.tolist()
            data4people.append(c_dict)
        data4pkl.append(data4people)

    with open(outputPklPath, "wb") as f:
        pickle.dump(data4pkl, f)

    return


def arrangeMMposePkl(poseInferencePklPath, outputPklPath):
    open_file = open(poseInferencePklPath, "rb")
    frames = pickle.load(open_file)
    open_file.close()

    markersMMpose = getMMposeMarkerNames()
    markersOpenPose = getOpenPoseMarkerNames()

    data4pkl = []
    for c_frame, frame in enumerate(frames):
        data4people = []
        for c, person in enumerate(frame):
            # coordinates = person["preds_with_flip"].tolist()
            coordinates = person["pred_instances"]["keypoints"][0, :, :]
            confidence = person["pred_instances"]["keypoint_scores"][0, :]
            # stack confidence with coordinates
            coordinates = np.column_stack((coordinates, confidence))
            c_coord_out = np.zeros((25 * 3,))
            for c_m, marker in enumerate(markersOpenPose):
                if marker == "midHip":
                    leftHip = coordinates[markersMMpose.index("LHip")]
                    rightHip = coordinates[markersMMpose.index("RHip")]
                    c_coord = []
                    # Mid point between both hips
                    c_coord.append((leftHip[0] + rightHip[0]) / 2)
                    c_coord.append((leftHip[1] + rightHip[1]) / 2)
                    # Lowest confidence
                    c_coord.append(np.min([leftHip[2], rightHip[2]]))
                elif marker == "Neck":
                    leftShoulder = coordinates[markersMMpose.index("LShoulder")]
                    rightShoulder = coordinates[markersMMpose.index("RShoulder")]
                    c_coord = []
                    # Mid point between both shoulders
                    c_coord.append((leftShoulder[0] + rightShoulder[0]) / 2)
                    c_coord.append((leftShoulder[1] + rightShoulder[1]) / 2)
                    # Lowest confidence
                    c_coord.append(np.min([leftShoulder[2], rightShoulder[2]]))
                else:
                    c_coord = coordinates[markersMMpose.index(marker)]
                idx_out = np.arange(c_m * 3, c_m * 3 + 3)
                c_coord_out[idx_out,] = c_coord
            c_dict = {}
            c_dict["person_id"] = [c]
            c_dict["pose_keypoints_2d"] = c_coord_out.tolist()
            data4people.append(c_dict)
        data4pkl.append(data4people)

    with open(outputPklPath, "wb") as f:
        pickle.dump(data4pkl, f)

    return


def getUpsampledMarkers(keypoints3D, frameRate):
    keypoints3D_res = np.empty(
        (keypoints3D.shape[2], keypoints3D.shape[0] * keypoints3D.shape[1])
    )
    for iFrame in range(keypoints3D.shape[2]):
        keypoints3D_res[iFrame, :] = np.reshape(
            keypoints3D[:, :, iFrame],
            (1, keypoints3D.shape[0] * keypoints3D.shape[1]),
            "F",
        )
    # Upsample to 100 Hz.
    newTime = np.arange(
        0, np.round(len(keypoints3D_res) / frameRate + 1 / 100, 6), 1 / 100
    )
    interpFxn = scipy.interpolate.interp1d(
        [i / frameRate for i in range(len(keypoints3D_res))],
        keypoints3D_res,
        axis=0,
        fill_value="extrapolate",
    )
    keypoints3D_res_interp = interpFxn(newTime)
    keypoints3D_interp = np.empty(
        (keypoints3D.shape[0], keypoints3D.shape[1], len(newTime))
    )
    for iFrame in range(len(newTime)):
        keypoints3D_interp[:, :, iFrame] = np.reshape(
            keypoints3D_res_interp[iFrame, :],
            (keypoints3D.shape[0], keypoints3D.shape[1]),
            "F",
        )

    return keypoints3D_interp
