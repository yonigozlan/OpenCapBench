import argparse
import os
import shutil

import yaml
from constants import config, config_global
from metrics import get_metrics
from pipeline import pipeline
from utils import importMetadata

import wandb


def main():
    parser = argparse.ArgumentParser(description="Benchmark OpenCap")
    parser.add_argument(
        "--dataDir",
        type=str,
        default=config["dataDir"],
        help="Directory where data is stored",
    )
    parser.add_argument(
        "--model_config_person",
        type=str,
        default=config["model_config_person"],
        help="Model config file for person detector",
    )
    parser.add_argument(
        "--model_ckpt_person",
        type=str,
        default=config["model_ckpt_person"],
        help="Model checkpoint file for person detector",
    )
    parser.add_argument(
        "--model_config_pose",
        type=str,
        default=config["model_config_pose"],
        help="Model config file for pose detector",
    )
    parser.add_argument(
        "--model_ckpt_pose",
        type=str,
        default=config["model_ckpt_pose"],
        help="Model checkpoint file for pose detector",
    )
    parser.add_argument(
        "--batch_size_det",
        type=int,
        default=config["batch_size_det"],
        help="Batch size for person detector",
    )
    parser.add_argument(
        "--batch_size_pose",
        type=int,
        default=config["batch_size_pose"],
        help="Batch size for pose detector",
    )
    parser.add_argument(
        "--dataName",
        type=str,
        default="Data",
        help="Name of data directory where predictions will be stored",
    )
    parser.add_argument(
        "--subjects", type=str, default="all", help="Subjects to process"
    )
    parser.add_argument(
        "--sessions", type=str, default="all", help="Sessions to process"
    )
    parser.add_argument(
        "--cameraSetups", type=str, default="2-cameras", help="Camera setups to process"
    )
    parser.add_argument(
        "--skip_pipeline",
        action="store_true",
        help="Skip pipeline and only run benchmark",
    )
    parser.add_argument(
        "--computeScale",
        action="store_true",
        default=False,
        help="Compute scale for each trial",
    )
    parser.add_argument(
        "--marker_set",
        type=str,
        default=config["marker_set"],
        help="Marker set to use for scaling",
    )
    parser.add_argument(
        "--alt_model",
        type=str,
        default=config["alt_model"],
        help="Alternative model to use for pose detector",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log metrics to wandb",
    )

    args = parser.parse_args()

    # replace config with args
    config["dataDir"] = args.dataDir
    config["model_config_person"] = args.model_config_person
    config["model_ckpt_person"] = args.model_ckpt_person
    config["model_config_pose"] = args.model_config_pose
    config["model_ckpt_pose"] = args.model_ckpt_pose
    if config_global == "local" or config_global == "windows":
        config["model_ckpt_pose_absolute"] = os.path.join(
            config["mmposeDirectory"], config["model_ckpt_pose"]
        )
    else:
        config["model_ckpt_pose_absolute"] = config["model_ckpt_pose"]
    config["batch_size_det"] = args.batch_size_det
    config["batch_size_pose"] = args.batch_size_pose
    config["dataName"] = args.dataName
    config["subjects"] = args.subjects
    config["sessions"] = args.sessions
    config["cameraSetups"] = args.cameraSetups
    config["skip_pipeline"] = args.skip_pipeline
    config["useGTscaling"] = not args.computeScale
    config["marker_set"] = args.marker_set
    config["alt_model"] = args.alt_model

    if not config["skip_pipeline"]:
        process_trials(config)

    dataDir = config["dataDir"]
    dataName = config["dataName"]

    mean_rmses, mean_rmses_no_shift = get_metrics(
        dataDir, os.path.join(dataDir, dataName)
    )

    print(type(mean_rmses))

    if args.wandb:
        wandb.init(project="opencap_bench", entity="yonigoz", name=config["dataName"])
        mean_RMSEs_table = wandb.Table(dataframe=mean_rmses)
        wandb.log({"Mean RMSEs": mean_RMSEs_table})


def process_trials(config):
    dataDir = config["dataDir"]
    useGTscaling = config["useGTscaling"]
    if config["marker_set"] != "Anatomical":
        useGTscaling = False

    # The dataset includes 2 sessions per subject.The first session includes
    # static, sit-to-stand, squat, and drop jump trials. The second session
    # includes walking trials. The sessions are named <subject_name>_Session0 and
    # <subject_name>_Session1.
    if config["subjects"] == "all":
        subjects = ["subject" + str(i) for i in range(2, 12)]
    else:
        subjects = [config["subjects"]]
    if config["sessions"] == "all":
        sessions = ["Session0", "Session1"]
    else:
        sessions = [config["sessions"]]
    sessionNames = [
        "{}_{}".format(subject, session) for subject in subjects for session in sessions
    ]

    poseDetectors = ["mmpose"]

    # Select the camera configuration you would like to use.
    # cameraSetups = ['2-cameras', '3-cameras', '5-cameras']
    cameraSetups = [config["cameraSetups"]]

    # Select the resolution at which you would like to use OpenPose. More details
    # about the options in Examples/reprocessSessions. In the paper, we compared
    # 'default' and '1x1008_4scales'.

    # To reprocess the data, we need to re-organize the data so that the folder
    # structure is the same one as the one expected by OpenCap. It is only done
    # once as long as the variable overwriteRestructuring is False. To overwrite
    # flip the flag to True.
    overwriteRestructuring = False
    for subject in subjects:
        pathSubject = os.path.join(dataDir, subject)
        pathVideos = os.path.join(pathSubject, "VideoData")
        for session in os.listdir(pathVideos):
            if "Session" not in session:
                continue
            pathSession = os.path.join(pathVideos, session)
            pathSessionNew = os.path.join(
                dataDir, config["dataName"], subject + "_" + session
            )
            if os.path.exists(pathSessionNew) and not overwriteRestructuring:
                continue
            os.makedirs(pathSessionNew, exist_ok=True)
            # Copy metadata
            pathMetadata = os.path.join(pathSubject, "sessionMetadata.yaml")
            shutil.copy2(pathMetadata, pathSessionNew)
            pathMetadataNew = os.path.join(pathSessionNew, "sessionMetadata.yaml")
            # Copy GT osim scaled model
            if useGTscaling:
                pathModel = os.path.join(
                    pathSubject,
                    "OpenSimData",
                    "Mocap",
                    "Model",
                    "LaiArnoldModified2017_poly_withArms_weldHand_scaled_adjusted.osim",
                )
                shutil.copy2(pathModel, pathSessionNew)
            # Adjust model name
            sessionMetadata = importMetadata(pathMetadataNew)
            sessionMetadata["openSimModel"] = "LaiUhlrich2022"
            with open(pathMetadataNew, "w") as file:
                yaml.dump(sessionMetadata, file)
            for cam in os.listdir(pathSession):
                if "Cam" not in cam:
                    continue
                pathCam = os.path.join(pathSession, cam)
                pathCamNew = os.path.join(pathSessionNew, "Videos", cam)
                pathInputMediaNew = os.path.join(pathCamNew, "InputMedia")
                # Copy videos.
                for trial in os.listdir(pathCam):
                    pathTrial = os.path.join(pathCam, trial)
                    if not os.path.isdir(pathTrial):
                        continue
                    suffix = "" if "extrinsics" in trial else "_syncdWithMocap"
                    pathVideo = os.path.join(pathTrial, trial + f"{suffix}.avi")
                    pathTrialNew = os.path.join(pathInputMediaNew, trial)
                    print("pathTrialNew", pathTrialNew)
                    os.makedirs(pathTrialNew, exist_ok=True)
                    shutil.copy2(pathVideo, os.path.join(pathTrialNew, trial + ".avi"))
                # Copy camera parameters
                pathParameters = os.path.join(
                    pathCam, "cameraIntrinsicsExtrinsics.pickle"
                )
                shutil.copy2(pathParameters, pathCamNew)

    # The dataset contains 5 videos per trial. The 5 videos are taken from cameras
    # positioned at different angles: Cam0:-70deg, Cam1:-45deg, Cam2:0deg,
    # Cam3:45deg, and Cam4:70deg where 0deg faces the participant. Depending on the
    # cameraSetup, we load different videos.
    cam2sUse = {
        "5-cameras": ["Cam0", "Cam1", "Cam2", "Cam3", "Cam4"],
        "3-cameras": ["Cam1", "Cam2", "Cam3"],
        "2-cameras": ["Cam1", "Cam3"],
    }

    for count, sessionName in enumerate(sessionNames):
        # Get trial names.
        pathCam0 = os.path.join(
            dataDir, config["dataName"], sessionName, "Videos", "Cam0", "InputMedia"
        )
        # Work around to re-order trials and have the extrinsics trial firs, and
        # the static second (if available).
        trials_tmp = os.listdir(pathCam0)
        trials_tmp = [t for t in trials_tmp if os.path.isdir(os.path.join(pathCam0, t))]
        session_with_static = False
        for trial in trials_tmp:
            if "extrinsics" in trial.lower():
                extrinsics_idx = trials_tmp.index(trial)
            if "static" in trial.lower():
                static_idx = trials_tmp.index(trial)
                session_with_static = True
        trials = [trials_tmp[extrinsics_idx]]
        if session_with_static:
            trials.append(trials_tmp[static_idx])
            for trial in trials_tmp:
                if "static" not in trial.lower() and "extrinsics" not in trial.lower():
                    trials.append(trial)
        else:
            for trial in trials_tmp:
                if "extrinsics" not in trial.lower():
                    trials.append(trial)

        for poseDetector in poseDetectors:
            for cameraSetup in cameraSetups:
                cam2Use = cam2sUse[cameraSetup]

                # The second sessions (<>_1) have no static trial for scaling the
                # model. The static trials were collected as part of the first
                # session for each subject (<>_0). We here copy the Model folder
                # from the first session to the second session.
                if sessionName[-1] == "1" and not useGTscaling:
                    sessionDir = os.path.join(dataDir, config["dataName"], sessionName)
                    sessionDir_0 = sessionDir[:-1] + "0"
                    camDir_0 = os.path.join(
                        sessionDir_0,
                        "OpenSimData",
                    )
                    modelDir_0 = os.path.join(camDir_0, "Model")
                    camDir_1 = os.path.join(
                        sessionDir,
                        "OpenSimData",
                    )
                    modelDir_1 = os.path.join(camDir_1, "Model")
                    os.makedirs(modelDir_1, exist_ok=True)
                    for file in os.listdir(modelDir_0):
                        pathFile = os.path.join(modelDir_0, file)
                        pathFileEnd = os.path.join(modelDir_1, file)
                        shutil.copy2(pathFile, pathFileEnd)

                # Process trial.
                for trial in trials:
                    print("Processing {}".format(trial))

                    # Detect if extrinsics trial to compute extrinsic parameters.
                    if "extrinsics" in trial.lower():
                        continue

                    # Detect if static trial with neutral pose to scale model.
                    if "static" in trial.lower():
                        scaleModel = True
                    else:
                        scaleModel = False

                    pipeline(
                        config,
                        sessionName=sessionName,
                        trialName=trial,
                        trial_id=trial,
                        camerasToUse=cam2Use,
                        poseDetector=poseDetector,
                        scaleModel=scaleModel,
                        useGTscaling=useGTscaling,
                    )


if __name__ == "__main__":
    main()
