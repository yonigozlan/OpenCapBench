import glob
import os
import pickle


def saveCameraParameters(filename, CameraParams):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    open_file = open(filename, "wb")
    pickle.dump(CameraParams, open_file)
    open_file.close()

    return True


def loadCameraParameters(filename):
    open_file = open(filename, "rb")
    cameraParams = pickle.load(open_file)

    open_file.close()
    return cameraParams


def getCameraParameters(sessionMetadata, sessionDir):
    # Get checkerboard parameters from metadata.
    CheckerBoardParams = {
        "dimensions": (
            sessionMetadata["checkerBoard"]["black2BlackCornersWidth_n"],
            sessionMetadata["checkerBoard"]["black2BlackCornersHeight_n"],
        ),
        "squareSize": sessionMetadata["checkerBoard"]["squareSideLength_mm"],
    }
    # Camera directories and models.
    cameraDirectories = {}
    cameraModels = {}
    for pathCam in glob.glob(os.path.join(sessionDir, "Videos", "Cam*")):
        if os.name == "nt":  # windows
            camName = pathCam.split("\\")[-1]
        elif os.name == "posix":  # ubuntu
            camName = pathCam.split("/")[-1]
        cameraDirectories[camName] = os.path.join(sessionDir, "Videos", pathCam)
        cameraModels[camName] = sessionMetadata["iphoneModel"][camName]

    # Get cameras' intrinsics and extrinsics.
    # Load parameters if saved, compute and save them if not.
    CamParamDict = {}
    loadedCamParams = {}
    for camName in cameraDirectories:
        camDir = cameraDirectories[camName]
        # Intrinsics ######################################################
        # Intrinsics and extrinsics already exist for this session.
        if os.path.exists(os.path.join(camDir, "cameraIntrinsicsExtrinsics.pickle")):
            print("Load extrinsics for {} - already existing".format(camName))
            CamParams = loadCameraParameters(
                os.path.join(camDir, "cameraIntrinsicsExtrinsics.pickle")
            )
            loadedCamParams[camName] = True

        # Extrinsics do not exist for this session.
        # else:
        #     print("Compute extrinsics for {} - not yet existing".format(camName))
        #     # Intrinsics ##################################################
        #     # Intrinsics directories.
        #     intrinsicDir = os.path.join(
        #         baseDir, "CameraIntrinsics", cameraModels[camName]
        #     )
        #     permIntrinsicDir = os.path.join(intrinsicDir, intrinsicsFinalFolder)
        #     # Intrinsics exist.
        #     if os.path.exists(permIntrinsicDir):
        #         CamParams = loadCameraParameters(
        #             os.path.join(permIntrinsicDir, "cameraIntrinsics.pickle")
        #         )
        #     # Intrinsics do not exist throw an error. Eventually the
        #     # webapp will give you the opportunity to compute them.

        #     else:
        #         exception = "Intrinsics don't exist for your camera model. OpenCap supports all iOS devices released in 2018 or later: https://www.opencap.ai/get-started."
        #         raise Exception(exception, exception)

        #     # Extrinsics ##################################################
        #     # Compute extrinsics from images popped out of this trial.
        #     # Hopefully you get a clean shot of the checkerboard in at
        #     # least one frame of each camera.
        #     useSecondExtrinsicsSolution = (
        #         alternateExtrinsics is not None and camName in alternateExtrinsics
        #     )
        #     pathVideoWithoutExtension = os.path.join(
        #         camDir, "InputMedia", trialName, trial_id
        #     )
        #     extension = getVideoExtension(pathVideoWithoutExtension)
        #     extrinsicPath = os.path.join(
        #         camDir, "InputMedia", trialName, trial_id + extension
        #     )

        #     # Modify intrinsics if camera view is rotated
        #     CamParams = rotateIntrinsics(CamParams, extrinsicPath)

        #     # for 720p, imageUpsampleFactor=4 is best for small board
        #     try:
        #         CamParams = calcExtrinsicsFromVideo(
        #             extrinsicPath,
        #             CamParams,
        #             CheckerBoardParams,
        #             visualize=False,
        #             imageUpsampleFactor=imageUpsampleFactor,
        #             useSecondExtrinsicsSolution=useSecondExtrinsicsSolution,
        #         )
        #     except Exception as e:
        #         if len(e.args) == 2:  # specific exception
        #             raise Exception(e.args[0], e.args[1])
        #         elif len(e.args) == 1:  # generic exception
        #             exception = "Camera calibration failed. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about camera calibration and https://www.opencap.ai/troubleshooting for potential causes for a failed calibration."
        #             raise Exception(exception, traceback.format_exc())
        #     loadedCamParams[camName] = False

        # Append camera parameters.
        if CamParams is not None:
            CamParamDict[camName] = CamParams.copy()
        else:
            CamParamDict[camName] = None

    # Save parameters if not existing yet.
    if not all([loadedCamParams[i] for i in loadedCamParams]):
        for camName in CamParamDict:
            saveCameraParameters(
                os.path.join(
                    cameraDirectories[camName], "cameraIntrinsicsExtrinsics.pickle"
                ),
                CamParamDict[camName],
            )

    return CamParamDict, cameraDirectories


def getCamAnglesOffsets(sessionMetadata, CamParamDict):
    # Detect if checkerboard is upside down.
    upsideDownChecker = isCheckerboardUpsideDown(CamParamDict)
    # Get rotation angles from motion capture environment to OpenSim.
    # Space-fixed are lowercase, Body-fixed are uppercase.
    checkerBoardMount = sessionMetadata["checkerBoard"]["placement"]
    if checkerBoardMount == "backWall" and not upsideDownChecker:
        rotationAngles = {"y": 90, "z": 180}
    elif checkerBoardMount == "backWall" and upsideDownChecker:
        rotationAngles = {"y": -90}
    elif checkerBoardMount == "backWall_largeCB":
        rotationAngles = {"y": -90}
    # TODO: uppercase?
    elif checkerBoardMount == "backWall_walking":
        rotationAngles = {"YZ": (-90, 180)}
    elif checkerBoardMount == "ground":
        rotationAngles = {"x": -90, "y": 90}
    elif checkerBoardMount == "ground_jumps":  # for sub1
        rotationAngles = {"x": 90, "y": 180}
    elif checkerBoardMount == "ground_gaits":  # for sub1
        rotationAngles = {"x": 90, "y": 90}
    else:
        raise Exception(
            "checkerBoard placement value in\
            sessionMetadata.yaml is not currently supported"
        )

    return rotationAngles


def isCheckerboardUpsideDown(CameraParams):
    # With backwall orientation, R[1,1] will always be positive in correct orientation
    # and negative if upside down
    for cam in list(CameraParams.keys()):
        if CameraParams[cam] is not None:
            upsideDown = CameraParams[cam]["rotation"][1, 1] < 0
            break
        # Default if no camera params (which is a garbage case anyway)
        upsideDown = False

    return upsideDown
