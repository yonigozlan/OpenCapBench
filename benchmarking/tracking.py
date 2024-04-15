import copy
import os
import pickle
import sys

import cv2
import numpy as np
from constants import getMMposeAnatomicalCocoMarkerNames
from scipy.signal import butter, sosfiltfilt
from utils import delete_multiple_element


def synchronizeVideos(
    CameraDirectories,
    trialRelativePath,
    pathPoseDetector,
    undistortPoints=False,
    CamParamDict=None,
    confidenceThreshold=0.3,
    filtFreqs={"gait": 12, "default": 30},
    imageBasedTracker=False,
    cams2Use=["all"],
    poseDetector="OpenPose",
    trialName=None,
    bbox_thr=0.8,
    marker_set="Anatomical",
):
    markerNames = getMMposeAnatomicalCocoMarkerNames()

    # Create list of cameras.
    if cams2Use[0] == "all":
        cameras2Use = list(CameraDirectories.keys())
    else:
        cameras2Use = cams2Use
    cameras2Use_in = copy.deepcopy(cameras2Use)

    # Initialize output lists
    pointList = []
    confList = []

    CameraDirectories_selectedCams = {}
    CamParamList_selectedCams = []
    for cam in cameras2Use:
        CameraDirectories_selectedCams[cam] = CameraDirectories[cam]
        CamParamList_selectedCams.append(CamParamDict[cam])

    # Load data.
    camsToExclude = []
    for camName in CameraDirectories_selectedCams:
        cameraDirectory = CameraDirectories_selectedCams[camName]
        videoFullPath = os.path.normpath(
            os.path.join(cameraDirectory, trialRelativePath)
        )
        trialPrefix, _ = os.path.splitext(os.path.basename(trialRelativePath))
        outputPklFolder = "OutputPkl_mmpose_" + str(bbox_thr)
        openposePklDir = os.path.join(outputPklFolder, trialName)
        pathOutputPkl = os.path.join(cameraDirectory, openposePklDir)
        ppPklPath = os.path.join(pathOutputPkl, trialPrefix + "_rotated_pp.pkl")
        key2D, confidence = loadPklVideo(
            ppPklPath,
            videoFullPath,
            imageBasedTracker=imageBasedTracker,
            poseDetector=poseDetector,
            marker_set=marker_set,
        )
        thisVideo = cv2.VideoCapture(videoFullPath[:-4] + "_rotated.avi")
        frameRate = np.round(thisVideo.get(cv2.CAP_PROP_FPS))
        if key2D.shape[1] == 0 and confidence.shape[1] == 0:
            camsToExclude.append(camName)
        else:
            # key2D, confidence = preprocess2Dkeypoints(key2D, confidence)
            pointList.append(key2D)
            confList.append(confidence)

    # If video is not existing, the corresponding camera should be removed.
    idx_camToExclude = []
    for camToExclude in camsToExclude:
        cameras2Use.remove(camToExclude)
        CameraDirectories_selectedCams.pop(camToExclude)
        idx_camToExclude.append(cameras2Use_in.index(camToExclude))
        # By removing the cameras in CamParamDict and CameraDirectories, we
        # modify those dicts in main, which is needed for the next stages.
        CamParamDict.pop(camToExclude)
        CameraDirectories.pop(camToExclude)
    delete_multiple_element(CamParamList_selectedCams, idx_camToExclude)

    # Synchronize keypoints.
    pointList, confList, nansInOutList, startEndFrameList = synchronizeVideoKeypoints(
        pointList,
        confList,
        confidenceThreshold=confidenceThreshold,
        filtFreqs=filtFreqs,
        sampleFreq=frameRate,
        visualize=False,
        maxShiftSteps=2 * frameRate,
        CameraParams=CamParamList_selectedCams,
        cameras2Use=cameras2Use,
        CameraDirectories=CameraDirectories_selectedCams,
        trialName=trialName,
    )

    if undistortPoints:
        if CamParamList_selectedCams is None:
            raise Exception("Need to have CamParamList to undistort Images")
        # nFrames = pointList[0].shape[1]
        unpackedPoints = unpackKeypointList(pointList)
        undistortedPoints = []
        for points in unpackedPoints:
            undistortedPoints.append(
                undistort2Dkeypoints(
                    points, CamParamList_selectedCams, useIntrinsicMatAsP=True
                )
            )
        pointList = repackKeypointList(undistortedPoints)

    pointDir = {}
    confDir = {}
    nansInOutDir = {}
    startEndFrames = {}
    for iCam, camName in enumerate(CameraDirectories_selectedCams):
        pointDir[camName] = pointList[iCam]
        confDir[camName] = confList[iCam]
        nansInOutDir[camName] = nansInOutList[iCam]
        startEndFrames[camName] = startEndFrameList[iCam]

    return (
        pointDir,
        confDir,
        markerNames,
        frameRate,
        nansInOutDir,
        startEndFrames,
        cameras2Use,
    )


def loadPklVideo(
    pklPath,
    videoFullPath,
    imageBasedTracker=False,
    poseDetector="OpenPose",
    marker_set="Anatomical",
):
    if marker_set == "Anatomical":
        # nb_keypoints = 53
        nb_keypoints = len(getMMposeAnatomicalCocoMarkerNames())
    else:
        nb_keypoints = 25
    open_file = open(pklPath, "rb")
    frames = pickle.load(open_file)
    open_file.close()

    nFrames = len(frames)

    # Read in JSON files.
    allPeople = []
    iPerson = 0
    anotherPerson = True

    while anotherPerson is True:
        anotherPerson = False
        res = np.zeros((nFrames, nb_keypoints * 3))
        res[:] = np.nan

        for c_frame, frame in enumerate(frames):
            # Get this persons keypoints if they exist.
            if len(frame) > iPerson:
                person = frame[iPerson]
                keypoints = person["pose_keypoints_2d"]
                res[c_frame, :] = keypoints

            # See if there are more people.
            if len(frame) > iPerson + 1:
                # If there was someone else in any frame, loop again.
                anotherPerson = True

        allPeople.append(res.copy())
        iPerson += 1

    # Track People, or if only one person, skip tracking
    if len(allPeople) > 1:
        # Select the largest keypoint-based bounding box as the subject of interest
        bbFromKeypoints = [keypointsToBoundingBox(data) for data in allPeople]
        maxArea, maxIdx = zip(
            *[
                getLargestBoundingBox(data, bbox)
                for data, bbox in zip(allPeople, bbFromKeypoints)
            ]
        )  # may want to find largest bounding box size in future instead of height

        # Check if a person has been detected, ie maxArea >= 0.0. If not, set
        # keypoints and confidence scores to 0, such that the camera is later
        # kicked out of the synchronization and triangulation.
        maxArea_np = np.array(maxArea)
        if np.max(maxArea_np) == 0.0:
            key2D = np.zeros((nb_keypoints, nFrames, 2))
            confidence = np.zeros((nb_keypoints, nFrames))
            return key2D, confidence

        startPerson = np.nanargmax(maxArea)
        startFrame = maxIdx[startPerson]
        startBb = bbFromKeypoints[startPerson][startFrame]

        # initialize output data
        res = np.zeros((nFrames, nb_keypoints * 3))
        # res[:] = np.nan
        # This just tracks the keypoint bounding box until there is a frame where the norm of the bbox corner change is above some
        # threshold (currently 20% of average image size). This percentage may need tuning

        # track this bounding box backwards until it can't be tracked
        res = trackKeypointBox(
            videoFullPath,
            startBb,
            allPeople,
            bbFromKeypoints,
            res,
            frameStart=startFrame,
            frameIncrement=-1,
            visualize=False,
            poseDetector=poseDetector,
        )

        # track this bounding box forward until it can't be tracked
        res = trackKeypointBox(
            videoFullPath,
            startBb,
            allPeople,
            bbFromKeypoints,
            res,
            frameStart=startFrame,
            frameIncrement=1,
            visualize=False,
            poseDetector=poseDetector,
        )
    else:
        res = allPeople[0]

    key2D = np.zeros((nb_keypoints, nFrames, 2))
    confidence = np.zeros((nb_keypoints, nFrames))
    for i in range(0, nb_keypoints):
        key2D[i, :, 0:2] = res[:, i * 3 : i * 3 + 2]
        confidence[i, :] = res[:, i * 3 + 2]

    # replace confidence nans with 0. 0 isn't used at all, nan is splined and used
    confidence = np.nan_to_num(confidence, nan=0)

    return key2D, confidence


def keypointsToBoundingBox(data):
    # input: nFrames x 75.
    # output: nFrames x 4 (xTopLeft, yTopLeft, width, height).

    c_data = np.copy(data)

    c_data[c_data == 0] = np.nan
    nonNanRows = np.argwhere(np.any(~np.isnan(c_data), axis=1))
    bbox = np.zeros((c_data.shape[0], 4))
    bbox[nonNanRows, 0] = np.nanmin(c_data[nonNanRows, 0::3], axis=2)
    bbox[nonNanRows, 1] = np.nanmin(c_data[nonNanRows, 1::3], axis=2)
    bbox[nonNanRows, 2] = np.nanmax(c_data[nonNanRows, 0::3], axis=2) - np.nanmin(
        c_data[nonNanRows, 0::3], axis=2
    )
    bbox[nonNanRows, 3] = np.nanmax(c_data[nonNanRows, 1::3], axis=2) - np.nanmin(
        c_data[nonNanRows, 1::3], axis=2
    )

    # Go a bit above head (this is for image-based tracker).
    bbox[:, 1] = np.maximum(0, bbox[:, 1] - 0.05 * bbox[:, 3])
    bbox[:, 3] = bbox[:, 3] * 1.05

    return bbox


def getLargestBoundingBox(data, bbox, confThresh=0.6):
    # Select the person/timepoint with the greatest bounding box area, with
    # reasonable mean confidence (i.e., closest to the camera).

    # Parameters (may require some tuning).
    # Don't consider frame if > this many keypoints with 0s or low confidence.
    nGoodKeypoints = 10
    # Threshold for low confidence keypoints.
    confThreshRemoveRow = 0.4

    # Copy data
    c_data = np.copy(data)
    c_data[c_data == 0] = np.nan
    conf = c_data[:, 2::3]
    c_bbox = np.copy(bbox)

    # Detect rows where < nGoodKeypoints markers have non-zeros.
    rows_nonzeros = np.count_nonzero(c_data, axis=1)
    rows_nonzeros_10m = np.argwhere(rows_nonzeros < nGoodKeypoints * 3)

    # Detect rows where < nGoodKeypoints markers have high confidence.
    nHighConfKeypoints = np.count_nonzero((conf > confThreshRemoveRow), axis=1)
    rows_lowConf = np.argwhere(nHighConfKeypoints < nGoodKeypoints)

    # Set bounding box to 0 for bad rows.
    badRows = np.unique(np.concatenate((rows_nonzeros_10m, rows_lowConf)))

    # Only remove rows if it isn't removing all rows
    if len(badRows) < c_data.shape[0]:
        c_bbox[badRows, :] = 0

    # Find bbox size.
    bbArea = np.multiply(c_bbox[:, 2], c_bbox[:, 3])

    # Find rows with high enough average confidence.
    confMask = np.zeros((conf.shape[0]), dtype=bool)
    nonNanRows = np.argwhere(np.any(~np.isnan(c_data), axis=1))
    confMask[nonNanRows] = np.nanmean(conf[nonNanRows, :], axis=2) > confThresh
    maskedArea = np.multiply(confMask, bbArea)
    maxArea = np.nanmax(maskedArea)
    try:
        idxMax = np.nanargmax(maskedArea)
    except:
        idxMax = np.nan

    return maxArea, idxMax


def trackKeypointBox(
    videoPath,
    bbStart,
    allPeople,
    allBoxes,
    dataOut,
    frameStart=0,
    frameIncrement=1,
    visualize=False,
    poseDetector="OpenPose",
    badFramesBeforeStop=0,
):
    # Tracks closest keypoint bounding boxes until the box changes too much.
    bboxKey = bbStart  # starting bounding box
    frameNum = frameStart

    # initiate video capture
    # Read video
    video = cv2.VideoCapture(videoPath)
    nFrames = allBoxes[0].shape[0]

    # Read desiredFrames.
    video.set(1, frameNum)
    ok, frame = video.read()
    if not ok:
        print("Cannot read video file")
        sys.exit()

    imageSize = (frame.shape[0], frame.shape[1])
    justStarted = True
    count = 0
    badFrames = []
    while frameNum > -1 and frameNum < nFrames:
        # Read a new frame

        if visualize:
            video.set(1, frameNum)
            ok, frame = video.read()
            if not ok:
                break

        # Find person closest to tracked bounding box, and fill their keypoint data
        keyBoxes = [box[frameNum] for box in allBoxes]

        # For OpenPose: a person is not always referred to by the same index,
        # so we identify the box that 'contains' the subject of interest. For
        # mmpose, a person is always referred to by the same index, since
        # persons are detected first and joints second. Here, we therefore
        # distinguish between both detection algorithms, and detect the person
        # for mmpose only for the first frame and then use the same person
        # index for the rest of the frames. For OpenPose, we detect the person
        # at each frame.
        if count == 0 or poseDetector == "OpenPose":
            iPerson, bboxKey_new, samePerson = findClosestBox(
                bboxKey, keyBoxes, imageSize
            )
        else:
            _, bboxKey_new, samePerson = findClosestBox(
                bboxKey, keyBoxes, imageSize, iPerson
            )

        # We allow badFramesBeforeStop of samePerson = False to account for an
        # errant frame(s) in the pose detector. Once we reach badFramesBeforeStop,
        # we break and output to the last good frame.
        if len(badFrames) > 0 and samePerson:
            badFrames = []

        if not samePerson and not justStarted:
            if len(badFrames) >= badFramesBeforeStop:
                print(
                    "not same person at "
                    + str(frameNum - frameIncrement * badFramesBeforeStop)
                )
                # Replace the data from the badFrames with zeros
                if len(badFrames) > 1:
                    dataOut[badFrames, :] = np.zeros(len(badFrames), dataOut.shape[0])
                break
            else:
                badFrames.append(frameNum)

        # Don't update the bboxKey for the badFrames
        if len(badFrames) == 0:
            bboxKey = bboxKey_new

        dataOut[frameNum, :] = allPeople[iPerson][frameNum, :]

        # Next frame
        frameNum += frameIncrement
        justStarted = False

        if visualize:
            p3 = (int(bboxKey[0]), int(bboxKey[1]))
            p4 = (int(bboxKey[0] + bboxKey[2]), int(bboxKey[1] + bboxKey[3]))
            cv2.rectangle(frame, p3, p4, (0, 255, 0), 2, 1)

            # Display result
            cv2.imshow("Tracking", frame)

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        count += 1

    return dataOut


def unpackKeypointList(keypointList):
    nFrames = keypointList[0].shape[1]
    unpackedKeypoints = []
    for iFrame in range(nFrames):
        tempList = []
        for keyArray in keypointList:
            tempList.append(keyArray[:, iFrame, None, :])
        unpackedKeypoints.append(tempList.copy())

    return unpackedKeypoints


def synchronizeVideoKeypoints(
    keypointList,
    confidenceList,
    confidenceThreshold=0.3,
    filtFreqs={"gait": 12, "default": 500},
    sampleFreq=30,
    visualize=False,
    maxShiftSteps=30,
    isGait=False,
    CameraParams=None,
    cameras2Use=["none"],
    CameraDirectories=None,
    trialName=None,
    trialID="",
    marker_set="Anatomical",
    bypassSync=True,
):
    # keypointList is a mCamera length list of (nmkrs,nTimesteps,2) arrays of camera 2D keypoints
    print("Synchronizing Keypoints")

    # Deep copies such that the inputs do not get modified.
    c_CameraParams = copy.deepcopy(CameraParams)
    c_cameras2Use = copy.deepcopy(cameras2Use)
    c_CameraDirectoryDict = copy.deepcopy(CameraDirectories)

    # Turn Camera Dict into List
    c_CameraDirectories = list(c_CameraDirectoryDict.values())
    # Check if one camera has only 0s as confidence scores, which would mean
    # no one has been properly identified. We want to kick out this camera
    # from the synchronization and triangulation. We do that by popping out
    # the corresponding data before syncing and add back 0s later.
    badCameras = []
    for icam, conf in enumerate(confidenceList):
        if np.max(conf) == 0.0:
            badCameras.append(icam)
    idxBadCameras = [badCameras[i] - i for i in range(len(badCameras))]
    for idxBadCamera in idxBadCameras:
        print("{} kicked out of synchronization".format(c_cameras2Use[idxBadCamera]))
        keypointList.pop(idxBadCamera)
        confidenceList.pop(idxBadCamera)
        c_CameraParams.pop(idxBadCamera)
        c_cameras2Use.pop(idxBadCamera)
        c_CameraDirectories.pop(idxBadCamera)

    # Select filtering frequency based on if it is gait
    filtFreq = filtFreqs["default"]

    # Filter keypoint data
    keyFiltList = []
    confFiltList = []
    confSyncFiltList = []
    nansInOutList = []
    for keyRaw, conf in zip(keypointList, confidenceList):
        keyRaw_clean = keyRaw
        conf_clean = conf
        nans_in_out = None
        conf_sync_clean = conf
        keyRaw_clean_filt = filterKeypointsButterworth(
            keyRaw_clean, filtFreq, sampleFreq, order=4
        )
        keyFiltList.append(keyRaw_clean_filt)
        confFiltList.append(conf_clean)
        confSyncFiltList.append(conf_sync_clean)
        nansInOutList.append(nans_in_out)

    # align signals - will start at the latest-starting frame (most negative shift) and end at
    # nFrames - the end of the earliest starting frame (nFrames - max shift)

    keypointsSync = []
    confidenceSync = []
    startEndFrames = []
    nansInOutSync = []
    if bypassSync:
        assert (
            len({len(key[0, :]) for key in keyFiltList}) == 1
        ), "All cameras must have same number of frames"
        for iCam, key in enumerate(keyFiltList):
            # Trim the keypoints and confidence lists
            confidence = confFiltList[iCam]
            iStart = 0
            iEnd = len(key[0, :])
            keypointsSync.append(key[:, iStart : iEnd + 1, :])
            confidenceSync.append(confidence[:, iStart : iEnd + 1])
            shiftednNansInOut = nansInOutList[iCam]
            nansInOutSync.append(shiftednNansInOut)
            # Save start and end frames to list, so can rewrite videos in
            # triangulateMultiviewVideo
            startEndFrames.append([iStart, iEnd])

    # We need to add back the cameras that have been kicked out.
    # We just add back zeros, they will be kicked out of the triangulation.
    for badCamera in badCameras:
        keypointsSync.insert(badCamera, np.zeros(keypointsSync[0].shape))
        confidenceSync.insert(badCamera, np.zeros(confidenceSync[0].shape))
        nansInOutSync.insert(badCamera, np.array([np.nan, np.nan]))
        startEndFrames.insert(badCamera, None)

    return keypointsSync, confidenceSync, nansInOutSync, startEndFrames


def undistort2Dkeypoints(pointList2D, CameraParamList, useIntrinsicMatAsP=True):
    # list of 2D points per image
    pointList2Dundistorted = []

    for i, points2D in enumerate(pointList2D):
        if useIntrinsicMatAsP:
            res = cv2.undistortPoints(
                points2D,
                CameraParamList[i]["intrinsicMat"],
                CameraParamList[i]["distortion"],
                P=CameraParamList[i]["intrinsicMat"],
            )
        else:
            res = cv2.undistortPoints(
                points2D,
                CameraParamList[i]["intrinsicMat"],
                CameraParamList[i]["distortion"],
            )
        pointList2Dundistorted.append(res.copy())

    return pointList2Dundistorted


def repackKeypointList(unpackedKeypointList):
    nFrames = len(unpackedKeypointList)
    nCams = len(unpackedKeypointList[0])
    nMkrs = unpackedKeypointList[0][0].shape[0]

    repackedKeypoints = []

    for iCam in range(nCams):
        tempArray = np.empty((nMkrs, nFrames, 2))
        for iFrame in range(nFrames):
            tempArray[:, iFrame, :] = np.squeeze(unpackedKeypointList[iFrame][iCam])
        repackedKeypoints.append(np.copy(tempArray))

    return repackedKeypoints


def findClosestBox(bbox, keyBoxes, imageSize, iPerson=None):
    # bbox: the bbox selected from the previous frame.
    # keyBoxes: bboxes detected in the current frame.
    # imageSize: size of the image
    # iPerson: index of the person to track..

    # Parameters.
    # Proportion of mean image dimensions that corners must change to be
    # considered different person
    cornerChangeThreshold = 0.2

    keyBoxCorners = []
    for keyBox in keyBoxes:
        keyBoxCorners.append(
            np.array(
                [keyBox[0], keyBox[1], keyBox[0] + keyBox[2], keyBox[1] + keyBox[3]]
            )
        )
    bboxCorners = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

    boxErrors = [np.linalg.norm(keyBox - bboxCorners) for keyBox in keyBoxCorners]
    try:
        if iPerson is None:
            iPerson = np.nanargmin(boxErrors)
        bbox = keyBoxes[iPerson]
    except:
        return None, None, False

    # If large jump in bounding box, break.
    samePerson = True
    if boxErrors[iPerson] > cornerChangeThreshold * np.mean(imageSize):
        samePerson = False

    return iPerson, bbox, samePerson


def filterKeypointsButterworth(key2D, filtFreq, sampleFreq, order=4):
    key2D_out = np.copy(key2D)
    wn = filtFreq / (sampleFreq / 2)
    if wn > 1:
        print(
            "You tried to filter "
            + str(int(sampleFreq))
            + " Hz signal with cutoff freq of "
            + str(int(filtFreq))
            + ". Will filter at "
            + str(int(sampleFreq / 2))
            + " instead."
        )
        wn = 0.99
    elif wn == 1:
        wn = 0.99

    sos = butter(order / 2, wn, btype="low", output="sos")

    for i in range(2):
        key2D_out[:, :, i] = sosfiltfilt(sos, key2D_out[:, :, i], axis=1)

    return key2D_out
