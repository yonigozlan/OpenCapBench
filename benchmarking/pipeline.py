import os
import traceback

from calibration import getCamAnglesOffsets, getCameraParameters
from kinematics import runIKTool
from pose import getUpsampledMarkers, runPoseDetector
from scaling import getScaleTimeRange, runScaleTool
from tracking import synchronizeVideos
from triangulation import triangulateMultiviewVideo
from utils import importMetadata
from utilsDataman import writeTRCfrom3DKeypoints


def pipeline(
    config,
    sessionName,
    trialName,
    trial_id,
    camerasToUse=["all"],
    poseDetector="mmpose",
    scaleModel=True,
    bbox_thr=0.8,
    benchmark=False,
    runUpsampling=True,
    useGTscaling=True,
):
    # High-level settings.
    # Camera calibration.
    runCameraCalibration = True
    # Pose detection.
    runPoseDetection = True
    # Video Synchronization.
    runSynchronization = True
    # Triangulation.
    runTriangulation = True
    # OpenSim pipeline.
    runOpenSimPipeline = True
    # Lowpass filter frequency of 2D keypoints for gait and everything else.
    filtFreqs = {"gait": 12, "default": 30}  # defaults to framerate/2
    # Set to False to only generate the json files (default is True).
    # This speeds things up and saves storage space.
    generateVideo = True

    # Paths and metadata. This gets defined through web app.
    baseDir = os.path.dirname(os.path.abspath(__file__))

    sessionDir = os.path.join(config["dataDir"], config["dataName"], sessionName)
    sessionMetadata = importMetadata(os.path.join(sessionDir, "sessionMetadata.yaml"))

    # Paths to pose detector folder for local testing.
    poseDetectorDirectory = config["mmposeDirectory"]

    # Camera calibration.
    if runCameraCalibration:
        CamParamDict, cameraDirectories = getCameraParameters(
            sessionMetadata, sessionDir
        )
        rotationAngles = getCamAnglesOffsets(sessionMetadata, CamParamDict)

    # 3D reconstruction
    # Create output folder.
    markerDir = os.path.join(sessionDir, "MarkerData")
    os.makedirs(markerDir, exist_ok=True)

    # Set output file name.
    pathOutputFiles = {}
    if benchmark:
        pathOutputFiles[trialName] = os.path.join(markerDir, trialName + ".trc")
    else:
        pathOutputFiles[trialName] = os.path.join(markerDir, trial_id + ".trc")

    # Trial relative path
    trialRelativePath = os.path.join("InputMedia", trialName, trial_id)

    if runPoseDetection:
        # Run pose detection algorithm.
        # try:
        runPoseDetector(
            config,
            cameraDirectories,
            trialRelativePath,
            poseDetectorDirectory,
            trialName,
            CamParamDict=CamParamDict,
            generateVideo=generateVideo,
            cams2Use=camerasToUse,
            poseDetector=poseDetector,
            bbox_thr=bbox_thr,
        )
        trialRelativePath += ".avi"

    if runSynchronization:
        # Synchronize videos.
        try:
            (
                keypoints2D,
                confidence,
                keypointNames,
                frameRate,
                nansInOut,
                startEndFrames,
                cameras2Use,
            ) = synchronizeVideos(
                cameraDirectories,
                trialRelativePath,
                poseDetectorDirectory,
                undistortPoints=True,
                CamParamDict=CamParamDict,
                filtFreqs=filtFreqs,
                confidenceThreshold=0.3,
                imageBasedTracker=False,
                cams2Use=camerasToUse,
                poseDetector=poseDetector,
                trialName=trialName,
                marker_set=config["marker_set"],
            )
        except Exception as e:
            if len(e.args) == 2:  # specific exception
                raise Exception(e.args[0], e.args[1])
            elif len(e.args) == 1:  # generic exception
                exception = """Video synchronization failed. Verify your setup and try again.
                    A fail-safe synchronization method is for the participant to
                    quickly raise one hand above their shoulders, then bring it back down.
                    Visit https://www.opencap.ai/best-pratices to learn more about
                    data collection and https://www.opencap.ai/troubleshooting for
                    potential causes for a failed trial."""
                raise Exception(exception, traceback.format_exc())

    if runTriangulation:
        # Triangulate.
        try:
            keypoints3D, confidence3D = triangulateMultiviewVideo(
                CamParamDict,
                keypoints2D,
                ignoreMissingMarkers=False,
                cams2Use=cameras2Use,
                confidenceDict=confidence,
                spline3dZeros=True,
                splineMaxFrames=int(frameRate / 5),
                nansInOut=nansInOut,
                CameraDirectories=cameraDirectories,
                trialName=trialName,
                startEndFrames=startEndFrames,
                trialID=trial_id,
            )
        except Exception as e:
            if len(e.args) == 2:  # specific exception
                raise Exception(e.args[0], e.args[1])
            elif len(e.args) == 1:  # generic exception
                exception = "Triangulation failed. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about data collection and https://www.opencap.ai/troubleshooting for potential causes for a failed trial."
                raise Exception(exception, traceback.format_exc())

        if runUpsampling:
            keypoints3D = getUpsampledMarkers(keypoints3D, frameRate)

        # Write TRC.
        writeTRCfrom3DKeypoints(
            keypoints3D,
            pathOutputFiles[trialName],
            keypointNames,
            frameRate=100,
            rotationAngles=rotationAngles,
        )

    # OpenSim pipeline.
    if runOpenSimPipeline:
        openSimPipelineDir = os.path.join(baseDir, "opensimPipeline")
        openSimDir = os.path.join(sessionDir, "OpenSimData")
        outputScaledModelDir = os.path.join(openSimDir, "Model")

        suffix_model = ""

        # Scaling.
        if scaleModel and not useGTscaling:
            os.makedirs(outputScaledModelDir, exist_ok=True)
            # Path setup file.
            if config["marker_set"] == "Anatomical":
                genericSetupFile4ScalingName = (
                    "Setup_scaling_RajagopalModified2016_withArms_KA_mmpose_final.xml"
                )
            elif config["marker_set"] == "Coco":
                genericSetupFile4ScalingName = (
                    "Setup_scaling_RajagopalModified2016_withArms_KA_mmpose.xml"
                )
            pathGenericSetupFile4Scaling = os.path.join(
                openSimPipelineDir, "Scaling", genericSetupFile4ScalingName
            )
            # Path model file.
            pathGenericModel4Scaling = os.path.join(
                openSimPipelineDir, "Models", sessionMetadata["openSimModel"] + ".osim"
            )
            # Path TRC file.
            pathTRCFile4Scaling = pathOutputFiles[trialName]
            # Get time range.
            try:
                timeRange4Scaling = getScaleTimeRange(
                    pathTRCFile4Scaling,
                    thresholdPosition=0.2,
                    thresholdTime=0.1,
                    removeRoot=True,
                    marker_set=config["marker_set"],
                    useWholeSeq=True,
                )
                # Run scale tool.
                print("Running Scaling")
                pathScaledModel = runScaleTool(
                    pathGenericSetupFile4Scaling,
                    pathGenericModel4Scaling,
                    sessionMetadata["mass_kg"],
                    pathTRCFile4Scaling,
                    timeRange4Scaling,
                    outputScaledModelDir,
                    subjectHeight=sessionMetadata["height_m"],
                    suffix_model=suffix_model,
                )
            except Exception as e:
                if len(e.args) == 2:  # specific exception
                    raise Exception(e.args[0], e.args[1])
                elif len(e.args) == 1:  # generic exception
                    exception = "Musculoskeletal model scaling failed. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about data collection and https://www.opencap.ai/troubleshooting for potential causes for a failed neutral pose."
                    raise Exception(exception, traceback.format_exc())

        # Inverse kinematics.
        if not scaleModel or useGTscaling:
            outputIKDir = os.path.join(openSimDir, "Kinematics")
            os.makedirs(outputIKDir, exist_ok=True)
            # Check if there is a scaled model.
            if useGTscaling:
                pathScaledModel = os.path.join(
                    sessionDir,
                    "LaiArnoldModified2017_poly_withArms_weldHand_scaled_adjusted.osim",
                )
            else:
                pathScaledModel = os.path.join(
                    outputScaledModelDir,
                    sessionMetadata["openSimModel"] + "_scaled.osim",
                )
            if os.path.exists(pathScaledModel):
                # Path setup file.
                if config["marker_set"] == "Anatomical":
                    genericSetupFile4IKName = "Setup_IK_mmpose_final_adjusted{}.xml".format(
                        suffix_model
                    )
                elif config["marker_set"] == "Coco":
                    genericSetupFile4IKName = "Setup_IK_mmpose{}.xml".format(
                        suffix_model
                    )
                pathGenericSetupFile4IK = os.path.join(
                    openSimPipelineDir, "IK", genericSetupFile4IKName
                )
                # Path TRC file.
                pathTRCFile4IK = pathOutputFiles[trialName]
                # Run IK tool.
                print("Running Inverse Kinematics")
                try:
                    runIKTool(
                        pathGenericSetupFile4IK,
                        pathScaledModel,
                        pathTRCFile4IK,
                        outputIKDir,
                    )
                except Exception as e:
                    if len(e.args) == 2:  # specific exception
                        raise Exception(e.args[0], e.args[1])
                    elif len(e.args) == 1:  # generic exception
                        exception = "Inverse kinematics failed. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about data collection and https://www.opencap.ai/troubleshooting for potential causes for a failed trial."
                        raise Exception(exception, traceback.format_exc())
            else:
                raise ValueError("No scaled model available.")
