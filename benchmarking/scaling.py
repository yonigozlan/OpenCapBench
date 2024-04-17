import os

import numpy as np
import opensim
import utilsDataman


def runScaleTool(
    pathGenericSetupFile,
    pathGenericModel,
    subjectMass,
    pathTRCFile,
    timeRange,
    pathOutputFolder,
    scaledModelName="not_specified",
    subjectHeight=0,
    createModelWithContacts=False,
    fixed_markers=False,
    suffix_model="",
):
    dirGenericModel, scaledModelNameA = os.path.split(pathGenericModel)

    # Paths.
    if scaledModelName == "not_specified":
        scaledModelName = scaledModelNameA[:-5] + "_scaled"
    pathOutputModel = os.path.join(pathOutputFolder, scaledModelName + ".osim")
    pathOutputMotion = os.path.join(pathOutputFolder, scaledModelName + ".mot")
    pathOutputSetup = os.path.join(
        pathOutputFolder, "Setup_Scale_" + scaledModelName + ".xml"
    )
    pathUpdGenericModel = os.path.join(
        pathOutputFolder, scaledModelNameA[:-5] + "_generic.osim"
    )

    # Marker set.
    _, setupFileName = os.path.split(pathGenericSetupFile)
    if "Lai" in scaledModelName or "Rajagopal" in scaledModelName:
        if "Mocap" in setupFileName:
            markerSetFileName = "RajagopalModified2016_markers_mocap{}.xml".format(
                suffix_model
            )
        elif "anatomical" in setupFileName or "final" in setupFileName:
            markerSetFileName = "RajagopalModified2016_markers_mmpose_final_adjusted.xml"
        elif "openpose" in setupFileName:
            markerSetFileName = "RajagopalModified2016_markers_openpose.xml"
        elif "mmpose" in setupFileName:
            markerSetFileName = "RajagopalModified2016_markers_mmpose.xml"
        else:
            if fixed_markers:
                markerSetFileName = "RajagopalModified2016_markers_augmenter_fixed.xml"
            else:
                markerSetFileName = (
                    "RajagopalModified2016_markers_augmenter{}.xml".format(suffix_model)
                )
    elif "gait2392" in scaledModelName:
        if "Mocap" in setupFileName:
            markerSetFileName = "gait2392_markers_mocap.xml"
        else:
            markerSetFileName = "gait2392_markers_augmenter.xml"
    else:
        raise ValueError("Unknown model type: scaling")
    pathMarkerSet = os.path.join(dirGenericModel, markerSetFileName)

    # Add the marker set to the generic model and save that updated model.
    opensim.Logger.setLevelString("error")
    genericModel = opensim.Model(pathGenericModel)
    markerSet = opensim.MarkerSet(pathMarkerSet)
    genericModel.set_MarkerSet(markerSet)
    genericModel.printToXML(pathUpdGenericModel)

    # Time range.
    timeRange_os = opensim.ArrayDouble(timeRange[0], 0)
    timeRange_os.insert(1, timeRange[-1])

    # Setup scale tool.
    scaleTool = opensim.ScaleTool(pathGenericSetupFile)
    scaleTool.setName(scaledModelName)
    scaleTool.setSubjectMass(subjectMass)
    scaleTool.setSubjectHeight(subjectHeight)
    genericModelMaker = scaleTool.getGenericModelMaker()
    genericModelMaker.setModelFileName(pathUpdGenericModel)
    modelScaler = scaleTool.getModelScaler()
    modelScaler.setMarkerFileName(pathTRCFile)
    modelScaler.setOutputModelFileName("")
    modelScaler.setOutputScaleFileName("")
    modelScaler.setTimeRange(timeRange_os)
    markerPlacer = scaleTool.getMarkerPlacer()
    markerPlacer.setMarkerFileName(pathTRCFile)
    markerPlacer.setOutputModelFileName(pathOutputModel)
    markerPlacer.setOutputMotionFileName(pathOutputMotion)
    markerPlacer.setOutputMarkerFileName("")
    markerPlacer.setTimeRange(timeRange_os)

    # Disable tasks of dofs that are locked and markers that are not present.
    model = opensim.Model(pathUpdGenericModel)
    coordNames = []
    for coord in model.getCoordinateSet():
        if not coord.getDefaultLocked():
            coordNames.append(coord.getName())
    modelMarkerNames = [marker.getName() for marker in model.getMarkerSet()]

    for task in markerPlacer.getIKTaskSet():
        # Remove IK tasks for dofs that are locked or don't exist.
        if (
            task.getName() not in coordNames
            and task.getConcreteClassName() == "IKCoordinateTask"
        ):
            task.setApply(False)
            print("{} is a locked coordinate - ignoring IK task".format(task.getName()))
        # Remove Marker tracking tasks for markers not in model.
        if (
            task.getName() not in modelMarkerNames
            and task.getConcreteClassName() == "IKMarkerTask"
        ):
            task.setApply(False)
            print("{} is not in model - ignoring IK task".format(task.getName()))

    # Remove measurements from measurement set when markers don't exist.
    # Disable entire measurement if no complete marker pairs exist.
    measurementSet = modelScaler.getMeasurementSet()
    for meas in measurementSet:
        mkrPairSet = meas.getMarkerPairSet()
        iMkrPair = 0
        while iMkrPair < meas.getNumMarkerPairs():
            mkrPairNames = [mkrPairSet.get(iMkrPair).getMarkerName(i) for i in range(2)]
            if any([mkr not in modelMarkerNames for mkr in mkrPairNames]):
                mkrPairSet.remove(iMkrPair)
                print(
                    "{} or {} not in model. Removing associated \
                      MarkerPairSet from {}.".format(
                        mkrPairNames[0], mkrPairNames[1], meas.getName()
                    )
                )
            else:
                iMkrPair += 1
            if meas.getNumMarkerPairs() == 0:
                meas.setApply(False)
                print(
                    "There were no marker pairs in {}, so this measurement \
                      is not applied.".format(
                        meas.getName()
                    )
                )
    # Run scale tool.
    scaleTool.printToXML(pathOutputSetup)
    command = "opensim-cmd -o error" + " run-tool " + pathOutputSetup
    os.system(command)

    # Sanity check
    scaled_model = opensim.Model(pathOutputModel)
    bodySet = scaled_model.getBodySet()
    nBodies = bodySet.getSize()
    scale_factors = np.zeros((nBodies, 3))
    for i in range(nBodies):
        bodyName = bodySet.get(i).getName()
        body = bodySet.get(bodyName)
        attached_geometry = body.get_attached_geometry(0)
        scale_factors[i, :] = attached_geometry.get_scale_factors().to_numpy()
    # print("scale factors: ", scale_factors)
    diff_scale = np.max(np.max(scale_factors, axis=0) - np.min(scale_factors, axis=0))
    # A difference in scaling factor larger than 1 would indicate that a
    # segment (e.g., humerus) would be more than twice as large as its generic
    # counterpart, whereas another segment (e.g., pelvis) would have the same
    # size as the generic segment. This is very unlikely, but might occur when
    # the camera calibration went wrong (i.e., bad extrinsics).
    if diff_scale > 1:
        exception = "Musculoskeletal model scaling failed; the segment sizes are not anthropometrically realistic. It is very likely that the camera calibration went wrong. Visit https://www.opencap.ai/best-pratices to learn more about camera calibration."
        raise Exception(exception, exception)

    return pathOutputModel


def getScaleTimeRange(
    pathTRCFile,
    thresholdPosition=0.05,
    thresholdTime=0.1,
    withArms=True,
    isMocap=False,
    marker_set="Anatomical",
    removeRoot=False,
    useWholeSeq=False,
):
    c_trc_file = utilsDataman.TRCFile(pathTRCFile)
    c_trc_time = c_trc_file.time
    if useWholeSeq:
        return [c_trc_time[0], c_trc_time[-1]]
    if marker_set == "Coco" or marker_set == "OpenPose":
        # No big toe markers, such as to include both OpenPose and mmpose.
        markers = [
            "Neck",
            "RShoulder",
            "LShoulder",
            "RHip",
            "LHip",
            "RKnee",
            "LKnee",
            "RAnkle",
            "LAnkle",
            "RHeel",
            "LHeel",
            "RSmallToe",
            "LSmallToe",
            "RElbow",
            "LElbow",
            "RWrist",
            "LWrist",
        ]
    elif marker_set == "Anatomical":
        markers = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "sternum",
            "rshoulder",
            "lshoulder",
            "r_lwrist",
            "l_lwrist",
            "r_mwrist",
            "l_mwrist",
            "r_ASIS",
            "l_ASIS",
            "r_PSIS",
            "l_PSIS",
            "r_ankle",
            "l_ankle",
            "r_mankle",
            "l_mankle",
            "r_5meta",
            "l_5meta",
            "r_big_toe",
            "l_big_toe",
            "l_calc",
            "r_calc",
            "C7",
            "L2",
            "T11",
            "T6",
        ]
    else:
        markers = [
            "C7_study",
            "r_shoulder_study",
            "L_shoulder_study",
            "r.ASIS_study",
            "L.ASIS_study",
            "r.PSIS_study",
            "L.PSIS_study",
            "r_knee_study",
            "L_knee_study",
            "r_mknee_study",
            "L_mknee_study",
            "r_ankle_study",
            "L_ankle_study",
            "r_mankle_study",
            "L_mankle_study",
            "r_calc_study",
            "L_calc_study",
            "r_toe_study",
            "L_toe_study",
            "r_5meta_study",
            "L_5meta_study",
            "RHJC_study",
            "LHJC_study",
        ]
        if withArms:
            markers.append("r_lelbow_study")
            markers.append("L_lelbow_study")
            markers.append("r_melbow_study")
            markers.append("L_melbow_study")
            markers.append("r_lwrist_study")
            markers.append("L_lwrist_study")
            markers.append("r_mwrist_study")
            markers.append("L_mwrist_study")

        if isMocap:
            markers = [marker.replace("_study", "") for marker in markers]
            markers = [
                marker.replace("r_shoulder", "R_Shoulder") for marker in markers
            ]  # should just change the mocap marker set
            markers = [
                marker.replace("L_shoulder", "L_Shoulder") for marker in markers
            ]  # should just change the mocap marker set
            markers = [
                marker.replace("RHJC", "R_HJC") for marker in markers
            ]  # should just change the mocap marker set
            markers = [
                marker.replace("LHJC", "L_HJC") for marker in markers
            ]  # should just change the mocap marker set
            markers = [
                marker.replace("r_lelbow", "R_elbow_lat") for marker in markers
            ]  # should just change the mocap marker set
            markers = [
                marker.replace("L_lelbow", "L_elbow_lat") for marker in markers
            ]  # should just change the mocap marker set
            markers = [
                marker.replace("r_melbow", "R_elbow_med") for marker in markers
            ]  # should just change the mocap marker set
            markers = [
                marker.replace("L_melbow", "L_elbow_med") for marker in markers
            ]  # should just change the mocap marker set
            markers = [
                marker.replace("r_lwrist", "R_wrist_radius") for marker in markers
            ]  # should just change the mocap marker set
            markers = [
                marker.replace("L_lwrist", "L_wrist_radius") for marker in markers
            ]  # should just change the mocap marker set
            markers = [
                marker.replace("r_mwrist", "R_wrist_ulna") for marker in markers
            ]  # should just change the mocap marker set
            markers = [
                marker.replace("L_mwrist", "L_wrist_ulna") for marker in markers
            ]  # should just change the mocap marker set

    trc_data = np.zeros((c_trc_time.shape[0], 3 * len(markers)))
    for count, marker in enumerate(markers):
        trc_data[:, count * 3 : count * 3 + 3] = c_trc_file.marker(marker)

    if removeRoot:
        try:
            root_data = c_trc_file.marker("midHip")
            trc_data -= np.tile(root_data, len(markers))
        except:
            pass

    if np.max(trc_data) > 10:  # in mm, turn to m
        trc_data /= 1000

    # Sampling frequency.
    sf = np.round(1 / np.mean(np.diff(c_trc_time)), 4)
    # Minimum duration for time range in seconds.
    timeRange_min = 1
    # Corresponding number of frames.
    nf = int(timeRange_min * sf + 1)

    detectedWindow = False
    i = 0
    while not detectedWindow:
        c_window = trc_data[i : i + nf, :]
        c_window_max = np.max(c_window, axis=0)
        c_window_min = np.min(c_window, axis=0)
        c_window_diff = np.abs(c_window_max - c_window_min)
        detectedWindow = np.alltrue(c_window_diff < thresholdPosition)
        if not detectedWindow:
            i += 1
            if i > c_trc_time.shape[0] - nf:
                i = 0
                nf -= int(0.1 * sf)
            if (
                np.round((nf - 1) / sf, 2) < thresholdTime
            ):  # number of frames got too small without detecting a window
                exception = (
                    "Musculoskeletal model scaling failed; could not detect a static phase of at least %.2fs. After you press record, make sure the subject stands still until the message tells you they can relax . Visit https://www.opencap.ai/best-pratices to learn more about data collection."
                    % thresholdTime
                )
                raise Exception(exception, exception)

    timeRange = [c_trc_time[i], c_trc_time[i + nf - 1]]
    timeRangeSpan = np.round(timeRange[1] - timeRange[0], 2)

    print(
        "Static phase of %.2fs detected in staticPose between [%.2f, %.2f]."
        % (timeRangeSpan, np.round(timeRange[0], 2), np.round(timeRange[1], 2))
    )

    return timeRange
