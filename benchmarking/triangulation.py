import copy
import glob
import os
import subprocess

import cv2
import numpy as np
from scipy.interpolate import pchip_interpolate
from utilsCameraPy3 import Camera, nview_linear_triangulations


def triangulateMultiviewVideo(
    CameraParamDict,
    keypointDict,
    imageScaleFactor=1,
    ignoreMissingMarkers=False,
    keypoints2D=[],
    cams2Use=["all"],
    confidenceDict={},
    trimTrial=False,
    spline3dZeros=False,
    splineMaxFrames=5,
    nansInOut=[],
    CameraDirectories=None,
    trialName=None,
    startEndFrames=None,
    trialID="",
):
    # cams2Use is a list of cameras that you want to use in triangulation.
    # if first entry of list is ['all'], will use all
    # otherwise, ['Cam0','Cam2']
    CameraParamList = [CameraParamDict[i] for i in CameraParamDict]
    if cams2Use[0] == "all" and not None in CameraParamList:
        keypointDict_selectedCams = keypointDict
        CameraParamDict_selectedCams = CameraParamDict
        confidenceDict_selectedCams = confidenceDict
    else:
        if cams2Use[0] == "all":  # must have been a none (uncalibrated camera)
            cams2Use = []
            for camName in CameraParamDict:
                if CameraParamDict[camName] is not None:
                    cams2Use.append(camName)

        keypointDict_selectedCams = {}
        CameraParamDict_selectedCams = {}
        if confidenceDict:
            confidenceDict_selectedCams = {}
        for camName in cams2Use:
            if CameraParamDict[camName] is not None:
                keypointDict_selectedCams[camName] = keypointDict[camName]
                CameraParamDict_selectedCams[camName] = CameraParamDict[camName]
                if confidenceDict:
                    confidenceDict_selectedCams[camName] = confidenceDict[camName]

    keypointList_selectedCams = [
        keypointDict_selectedCams[i] for i in keypointDict_selectedCams
    ]
    confidenceList_selectedCams = [
        confidenceDict_selectedCams[i] for i in confidenceDict_selectedCams
    ]
    CameraParamList_selectedCams = [
        CameraParamDict_selectedCams[i] for i in CameraParamDict_selectedCams
    ]
    unpackedKeypoints = unpackKeypointList(keypointList_selectedCams)
    points3D = np.zeros(
        (
            3,
            keypointList_selectedCams[0].shape[0],
            keypointList_selectedCams[0].shape[1],
        )
    )
    confidence3D = np.zeros(
        (
            1,
            keypointList_selectedCams[0].shape[0],
            keypointList_selectedCams[0].shape[1],
        )
    )

    for iFrame, points2d in enumerate(unpackedKeypoints):
        # If confidence weighting
        if confidenceDict:
            thisConfidence = [c[:, iFrame] for c in confidenceList_selectedCams]
        else:
            thisConfidence = None

        points3D[:, :, iFrame], confidence3D[:, :, iFrame] = triangulateMultiview(
            CameraParamList_selectedCams,
            points2d,
            imageScaleFactor=1,
            useRotationEuler=False,
            ignoreMissingMarkers=ignoreMissingMarkers,
            keypoints2D=keypoints2D,
            confidence=thisConfidence,
        )

    startInd = 0
    endInd = confidence3D.shape[2]

    # Rewrite videos based on sync time and trimmed trc.
    if CameraDirectories != None and trialName != None:
        print("Writing synchronized videos")
        outputVideoDir = os.path.abspath(
            os.path.join(
                list(CameraDirectories.values())[0],
                "../../",
                "VisualizerVideos",
                trialName,
            )
        )
        os.makedirs(outputVideoDir, exist_ok=True)
        for iCam, camName in enumerate(keypointDict):
            nFramesToWrite = endInd - startInd

            inputPaths = glob.glob(
                os.path.join(
                    CameraDirectories[camName], "OutputMedia*", trialName, trialID + "*"
                )
            )
            if len(inputPaths) > 0:
                inputPath = inputPaths[0]
            else:
                inputPaths = glob.glob(
                    os.path.join(
                        CameraDirectories[camName],
                        "InputMedia*",
                        trialName,
                        trialID + "*",
                    )
                )
                inputPath = inputPaths[0]

            # get frame rate and assume all the same for sync'd videos
            if iCam == 0:
                thisVideo = cv2.VideoCapture(inputPath)
                frameRate = np.round(thisVideo.get(cv2.CAP_PROP_FPS))
                thisVideo.release()

            # Only rewrite if camera in cams2use and wasn't kicked out earlier
            if (camName in cams2Use or cams2Use[0] == "all") and startEndFrames[
                camName
            ] != None:
                _, inputName = os.path.split(inputPath)
                inputRoot, inputExt = os.path.splitext(inputName)

                # Let's use mp4 since we write for the internet
                outputFileName = inputRoot + "_syncd_" + camName + ".mp4 "  # inputExt

                thisStartFrame = startInd + startEndFrames[camName][0]

                rewriteVideos(
                    inputPath,
                    thisStartFrame,
                    nFramesToWrite,
                    frameRate,
                    outputDir=outputVideoDir,
                    imageScaleFactor=0.5,
                    outputFileName=outputFileName,
                )

    if spline3dZeros:
        # Spline across positions with 0 3D confidence (i.e., there weren't 2 cameras
        # to use for triangulation).
        points3D = spline3dPoints(points3D, confidence3D, splineMaxFrames)

    return points3D, confidence3D


def triangulateMultiview(
    CameraParamList,
    points2dUndistorted,
    imageScaleFactor=1,
    useRotationEuler=False,
    ignoreMissingMarkers=False,
    selectCamerasMinReprojError=False,
    ransac=False,
    keypoints2D=[],
    confidence=None,
):
    # create a list of cameras (says sequence in documentation) from CameraParamList
    cameraList = []

    for camParams in CameraParamList:
        # get rotation matrix
        if useRotationEuler:
            rotMat = cv2.Rodrigues(camParams["rotation_EulerAngles"])[0]
        else:
            rotMat = camParams["rotation"]

        c = Camera()
        c.set_K(camParams["intrinsicMat"])
        c.set_R(rotMat)
        c.set_t(np.reshape(camParams["translation"], (3, 1)))
        cameraList.append(c)

    # triangulate
    stackedPoints = np.stack(points2dUndistorted)
    pointsInput = []
    for i in range(stackedPoints.shape[1]):
        pointsInput.append(stackedPoints[:, i, 0, :].T)

    points3d, confidence3d = nview_linear_triangulations(
        cameraList, pointsInput, weights=confidence
    )

    return points3d, confidence3d


def spline3dPoints(points3D, confidence3D, splineMaxFrames=5):
    c_p3d = copy.deepcopy(points3D)
    c_conf = copy.deepcopy(confidence3D)

    # Find internal stretches of 0 that are shorter than splineMaxFrames
    for iPt in np.arange(c_p3d.shape[1]):
        thisConf = c_conf[0, iPt, :]
        zeroInds, nonZeroInds = findInternalZeroInds(thisConf, splineMaxFrames)

        # spline these internal zero stretches
        if zeroInds is not None and zeroInds.shape[0] > 0:
            c_p3d[:, iPt, zeroInds] = pchip_interpolate(
                nonZeroInds, c_p3d[:, iPt, nonZeroInds], zeroInds, axis=1
            )

    return c_p3d


def unpackKeypointList(keypointList):
    nFrames = keypointList[0].shape[1]
    unpackedKeypoints = []
    for iFrame in range(nFrames):
        tempList = []
        for keyArray in keypointList:
            tempList.append(keyArray[:, iFrame, None, :])
        unpackedKeypoints.append(tempList.copy())

    return unpackedKeypoints


def findInternalZeroInds(x, maxLength):
    # skip splining if x is all 0
    if all(x == 0):
        return None, None

    zeroInds = np.argwhere(x == 0)
    dZeroInds = np.diff(zeroInds, axis=0, prepend=-1)

    # check if first/last values are 0, don't spline the beginning and end
    start0 = x[0] == 0
    end0 = x[-1] == 0

    if start0:
        # delete string of 0s at beginning
        while dZeroInds.shape[0] > 0 and dZeroInds[0] == 1:
            zeroInds = np.delete(zeroInds, 0)
            dZeroInds = np.delete(dZeroInds, 0)
    if end0:
        while dZeroInds.shape[0] > 0 and dZeroInds[-1] == 1:
            # keep deleting end inds if there is a string of 0s before end
            zeroInds = np.delete(zeroInds, -1)
            dZeroInds = np.delete(dZeroInds, -1)
        # delete last index before jump - value will be greater than 1
        zeroInds = np.delete(zeroInds, -1)
        dZeroInds = np.delete(dZeroInds, -1)

    # check if any stretches are longer than maxLength
    thisStretch = np.array([0])  # initialize with first value b/c dZeroInds[0] ~=1
    indsToDelete = np.array([])
    for iIdx, d in enumerate(dZeroInds):
        if d == 1:
            thisStretch = np.append(thisStretch, iIdx)
        else:
            if len(thisStretch) >= maxLength:
                indsToDelete = np.append(indsToDelete, np.copy(thisStretch))
            thisStretch = np.array([iIdx])
    # if final stretch is too long, add it too
    if len(thisStretch) >= maxLength:
        indsToDelete = np.append(indsToDelete, np.copy(thisStretch))

    if len(indsToDelete) > 0:
        zeroInds = np.delete(zeroInds, indsToDelete.astype(int))
        dZeroInds = np.delete(dZeroInds, indsToDelete.astype(int))

    nonZeroInds = np.delete(np.arange(len(x)), zeroInds)

    return zeroInds, nonZeroInds


def rewriteVideos(
    inputPath,
    startFrame,
    nFrames,
    frameRate,
    outputDir=None,
    imageScaleFactor=0.5,
    outputFileName=None,
):
    inputDir, vidName = os.path.split(inputPath)
    vidName, vidExt = os.path.splitext(vidName)

    if outputFileName is None:
        outputFileName = vidName + "_sync" + vidExt
    if outputDir is not None:
        outputFullPath = os.path.join(outputDir, outputFileName)
    else:
        outputFullPath = os.path.join(inputDir, outputFileName)

    imageScaleArg = ""  # None if want to keep image size the same
    maintainQualityArg = "-acodec copy -vcodec copy"
    if imageScaleFactor is not None:
        imageScaleArg = "-vf scale=iw/{:.0f}:-1".format(1 / imageScaleFactor)
        maintainQualityArg = ""

    startTime = startFrame / frameRate

    # We need to replace double space to single space for split to work
    # That's a bit hacky but works for now. (TODO)
    ffmpegCmd = (
        "ffmpeg -loglevel error -y -ss {:.3f} -i {} {} -vframes {:.0f} {} {}".format(
            startTime,
            inputPath,
            maintainQualityArg,
            nFrames,
            imageScaleArg,
            outputFullPath,
        )
        .rstrip()
        .replace("  ", " ")
    )

    subprocess.run(ffmpegCmd.split(" "))

    return
