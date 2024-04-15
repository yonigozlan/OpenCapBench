import argparse
import glob
import json
import os

import pandas as pd


def mot_to_df(motPath):
    # parse the mocap motion file
    with open(motPath, "r") as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if line.startswith("endheader"):
                break
        data = lines[i + 1 :]

    # parse data into table
    data = [x.split() for x in data]
    df = pd.DataFrame(data[1:], columns=data[0])
    df = df.apply(pd.to_numeric, errors="ignore")
    df = df.set_index("time")

    return df


def build_exercises_dict(gt_path, pred_path):
    # get all mot files in the directory
    subjects_dict = {}
    subjects_dir = glob.glob(os.path.join(gt_path,"subject*"))
    for dir in subjects_dir:
        motFiles = glob.glob(os.path.join(dir,"OpenSimData/Mocap/IK/*.mot"))
        subject_key = dir.split(os.sep)[-1]
        for motFile in motFiles:
            if subject_key not in subjects_dict:
                subjects_dict[subject_key] = {motFile.split(os.sep)[-1]: {"gt": motFile}}
            else:
                subjects_dict[subject_key][motFile.split(os.sep)[-1]] = {"gt": motFile}

    predicted_subjects_dir = glob.glob(os.path.join(pred_path, "subject*"))
    for dir in predicted_subjects_dir:
        motFiles = glob.glob(os.path.join(dir, "OpenSimData/Kinematics/*.mot"))
        subject_key = dir.split(os.sep)[-1].split("_")[0]
        for motFile in motFiles:
            if motFile.split(os.sep)[-1] in subjects_dict[subject_key]:
                subjects_dict[subject_key][motFile.split(os.sep)[-1]][
                    "predicted"
                ] = motFile

    return subjects_dict


def get_best_shifts_for_metric(exercises_dict, metric):
    best_shifts = {subject: {} for subject in exercises_dict}
    df_dict = {subject: {} for subject in exercises_dict}
    for subject in exercises_dict:
        df_dict[subject] = {motFile: {} for motFile in exercises_dict[subject]}
        for motFile in exercises_dict[subject]:
            if "predicted" in exercises_dict[subject][motFile]:
                mocapMotPath_gt = exercises_dict[subject][motFile]["gt"]
                mocapMotPath = exercises_dict[subject][motFile]["predicted"]
                df_gt = mot_to_df(mocapMotPath_gt)
                df = mot_to_df(mocapMotPath)

                df_dict[subject][motFile]["gt"] = df_gt
                df_dict[subject][motFile]["predicted"] = df

                lowest_mse = 100000
                best_shift = 0
                for shift in range(-40, 40):
                    df_shifted = df.shift(shift)

                    df_diff = df_shifted - df_gt
                    df_diff = df_diff.dropna()
                    mse = df_diff[metric].abs().mean()
                    if mse < lowest_mse:
                        lowest_mse = mse
                        best_shift = shift
                best_shifts[subject][motFile] = (best_shift, lowest_mse)

    return best_shifts, df_dict


def compute_median_shifts(best_shifts_list, df_dict):
    median_shifts = {subject: {} for subject in best_shifts_list[0]}
    for subject in best_shifts_list[0]:
        for motFile in best_shifts_list[0][subject]:
            shifts = [
                best_shift[subject][motFile][0] for best_shift in best_shifts_list
            ]
            median_shifts[subject][motFile] = [int(sum(shifts) / len(shifts))]
            df = df_dict[subject][motFile]["predicted"]
            df_gt = df_dict[subject][motFile]["gt"]
            df_shifted = df.shift(median_shifts[subject][motFile][0])
            # compute number of frames overlapping before and after the shift
            median_shifts[subject][motFile].append(
                (df_shifted - df_gt).dropna().shape[0]
            )
            median_shifts[subject][motFile].append((df - df_gt).dropna().shape[0])

    return median_shifts


def get_metrics(gt_dir, pred_dir, output_dir=None):
    exercises_dict = build_exercises_dict(gt_dir, pred_dir)
    best_shifts_r, df_dict = get_best_shifts_for_metric(exercises_dict, "knee_angle_r")
    best_shifts_l, df_dict = get_best_shifts_for_metric(exercises_dict, "knee_angle_l")
    best_shifts = [best_shifts_r, best_shifts_l]
    best_shifts = [best_shifts_r, best_shifts_l]
    median_shifts = compute_median_shifts(best_shifts, df_dict)

    if output_dir is None:
        output_dir = pred_dir
    with open(os.path.join(output_dir, "shifts.json"), "w") as file:
        # format json
        json.dump(median_shifts, file, indent=4)

    # compute mean rmse for each metric
    rmses = {subject: {} for subject in exercises_dict}
    for subject in exercises_dict:
        rmses[subject] = {motFile: None for motFile in exercises_dict[subject]}
        for motFile in exercises_dict[subject]:
            if "predicted" in exercises_dict[subject][motFile]:
                mocapMotPath_gt = exercises_dict[subject][motFile]["gt"]
                mocapMotPath = exercises_dict[subject][motFile]["predicted"]
                df_gt = mot_to_df(mocapMotPath_gt)
                df = mot_to_df(mocapMotPath)
                df_shifted = df.shift(median_shifts[subject][motFile][0])
                df_diff_squared = (df_shifted - df_gt).pow(2)
                df_diff_squared = df_diff_squared.dropna()
                df_diff_squared = df_diff_squared[
                    df_diff_squared.columns[
                        : df_diff_squared.columns.get_loc("lumbar_rotation") + 1
                    ]
                ]
                df_diff_squared = df_diff_squared.drop(
                    [
                        "pelvis_tx",
                        "pelvis_ty",
                        "pelvis_tz",
                        "knee_angle_l_beta",
                        "knee_angle_r_beta",
                        "mtp_angle_l",
                        "mtp_angle_r",
                    ],
                    axis=1,
                )
                rmses[subject][motFile] = df_diff_squared.mean().pow(0.5)

    # compute mean df for all rmses
    list_of_rmses = [
        rmses[subject][motFile]
        for subject in rmses
        for motFile in rmses[subject]
        if rmses[subject][motFile] is not None
    ]
    concatenated_rmses = pd.concat(list_of_rmses)
    grouped = concatenated_rmses.groupby(level=0)
    mean_rmses = grouped.mean()
    # get columns of panda series
    for col in mean_rmses.index.to_list():
        if col.endswith("_r"):
            mean_rmses[col[:-2]] = (mean_rmses[col] + mean_rmses[col[:-2] + "_l"]) / 2
            mean_rmses = mean_rmses.drop(col)
            mean_rmses = mean_rmses.drop(col[:-2] + "_l")
    # only keep 3 significant digits
    mean_rmses = mean_rmses.round(3)
    # compute mean of columns which finish by _r and _l

    print("mean rmses: ", mean_rmses)
    # export to csv
    mean_rmses.to_csv(
        os.path.join(output_dir, pred_dir.split(os.sep)[-1] + "mean_rmses.csv")
    )

    # same but without the median shift
    # compute mean rmse for each metric
    rmses = {subject: {} for subject in exercises_dict}
    for subject in exercises_dict:
        rmses[subject] = {motFile: None for motFile in exercises_dict[subject]}
        for motFile in exercises_dict[subject]:
            if "predicted" in exercises_dict[subject][motFile]:
                mocapMotPath_gt = exercises_dict[subject][motFile]["gt"]
                mocapMotPath = exercises_dict[subject][motFile]["predicted"]
                df_gt = mot_to_df(mocapMotPath_gt)
                df = mot_to_df(mocapMotPath)
                df_diff_squared = (df - df_gt).pow(2)
                df_diff_squared = df_diff_squared.dropna()
                df_diff_squared = df_diff_squared[
                    df_diff_squared.columns[
                        : df_diff_squared.columns.get_loc("lumbar_rotation") + 1
                    ]
                ]
                df_diff_squared = df_diff_squared.drop(
                    [
                        "pelvis_tx",
                        "pelvis_ty",
                        "pelvis_tz",
                        "knee_angle_l_beta",
                        "knee_angle_r_beta",
                        "mtp_angle_l",
                        "mtp_angle_r",
                    ],
                    axis=1,
                )
                rmses[subject][motFile] = df_diff_squared.mean().pow(0.5)

    # compute mean df for all rmses
    list_of_rmses = [
        rmses[subject][motFile]
        for subject in rmses
        for motFile in rmses[subject]
        if rmses[subject][motFile] is not None
    ]
    concatenated_rmses = pd.concat(list_of_rmses)
    grouped = concatenated_rmses.groupby(level=0)
    mean_rmses_no_shift = grouped.mean()
    for col in mean_rmses_no_shift.index:
        if col.endswith("_r"):
            mean_rmses_no_shift[col[:-2]] = (
                mean_rmses_no_shift[col] + mean_rmses_no_shift[col[:-2] + "_l"]
            ) / 2
            mean_rmses_no_shift = mean_rmses_no_shift.drop(col)
            mean_rmses_no_shift = mean_rmses_no_shift.drop(col[:-2] + "_l")
    # only keep 3 significant digits

    mean_rmses_no_shift = mean_rmses_no_shift.round(3)

    print("mean_rmses_no_shift: ", mean_rmses_no_shift)
    # export to csv
    mean_rmses_no_shift.to_csv(
        os.path.join(output_dir, pred_dir.split(os.sep)[-1] + "mean_rmses_no_shift.csv")
    )

    return mean_rmses, mean_rmses_no_shift


if __name__ == "__main__":
    gt_dir = None
    parser = argparse.ArgumentParser(description="Correct metrics for OpenSim")
    parser.add_argument("pred_dir", type=str, help="Path to the prediction directory")
    parser.add_argument(
        "--output", type=str, help="Path to the output directory", default=None
    )

    args = parser.parse_args()
    if args.output is None:
        args.output = args.pred_dir

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    exercises_dict = build_exercises_dict(gt_dir, args.pred_dir)
    best_shifts_r, df_dict = get_best_shifts_for_metric(exercises_dict, "knee_angle_r")
    best_shifts_l, df_dict = get_best_shifts_for_metric(exercises_dict, "knee_angle_l")
    best_shifts = [best_shifts_r, best_shifts_l]
    best_shifts = [best_shifts_r, best_shifts_l]
    median_shifts = compute_median_shifts(best_shifts, df_dict)

    with open(os.path.join(args.output, "shifts.json"), "w") as file:
        # format json
        json.dump(median_shifts, file, indent=4)

    # compute mean rmse for each metric
    rmses = {subject: {} for subject in exercises_dict}
    for subject in exercises_dict:
        rmses[subject] = {motFile: None for motFile in exercises_dict[subject]}
        for motFile in exercises_dict[subject]:
            if "predicted" in exercises_dict[subject][motFile]:
                mocapMotPath_gt = exercises_dict[subject][motFile]["gt"]
                mocapMotPath = exercises_dict[subject][motFile]["predicted"]
                df_gt = mot_to_df(mocapMotPath_gt)
                df = mot_to_df(mocapMotPath)
                df_shifted = df.shift(median_shifts[subject][motFile][0])
                df_diff_squared = (df_shifted - df_gt).pow(2)
                df_diff_squared = df_diff_squared.dropna()
                df_diff_squared = df_diff_squared[
                    df_diff_squared.columns[
                        : df_diff_squared.columns.get_loc("lumbar_rotation") + 1
                    ]
                ]
                df_diff_squared = df_diff_squared.drop(
                    [
                        "pelvis_tx",
                        "pelvis_ty",
                        "pelvis_tz",
                        "knee_angle_l_beta",
                        "knee_angle_r_beta",
                        "mtp_angle_l",
                        "mtp_angle_r",
                    ],
                    axis=1,
                )
                rmses[subject][motFile] = df_diff_squared.mean().pow(0.5)

    # compute mean df for all rmses
    list_of_rmses = [
        rmses[subject][motFile]
        for subject in rmses
        for motFile in rmses[subject]
        if rmses[subject][motFile] is not None
    ]
    concatenated_rmses = pd.concat(list_of_rmses)
    grouped = concatenated_rmses.groupby(level=0)
    mean_rmses = grouped.mean()
    # get columns of panda series
    for col in mean_rmses.index.to_list():
        if col.endswith("_r"):
            mean_rmses[col[:-2]] = (mean_rmses[col] + mean_rmses[col[:-2] + "_l"]) / 2
            mean_rmses = mean_rmses.drop(col)
            mean_rmses = mean_rmses.drop(col[:-2] + "_l")
    # only keep 3 significant digits
    mean_rmses = mean_rmses.round(3)
    # compute mean of columns which finish by _r and _l

    print("mean rmses: ", mean_rmses)
    # export to csv
    mean_rmses.to_csv(
        os.path.join(args.output, args.pred_dir.split("/")[-1] + "mean_rmses.csv")
    )

    # same but without the median shift
    # compute mean rmse for each metric
    rmses = {subject: {} for subject in exercises_dict}
    for subject in exercises_dict:
        rmses[subject] = {motFile: None for motFile in exercises_dict[subject]}
        for motFile in exercises_dict[subject]:
            if "predicted" in exercises_dict[subject][motFile]:
                mocapMotPath_gt = exercises_dict[subject][motFile]["gt"]
                mocapMotPath = exercises_dict[subject][motFile]["predicted"]
                df_gt = mot_to_df(mocapMotPath_gt)
                df = mot_to_df(mocapMotPath)
                df_diff_squared = (df - df_gt).pow(2)
                df_diff_squared = df_diff_squared.dropna()
                df_diff_squared = df_diff_squared[
                    df_diff_squared.columns[
                        : df_diff_squared.columns.get_loc("lumbar_rotation") + 1
                    ]
                ]
                df_diff_squared = df_diff_squared.drop(
                    [
                        "pelvis_tx",
                        "pelvis_ty",
                        "pelvis_tz",
                        "knee_angle_l_beta",
                        "knee_angle_r_beta",
                        "mtp_angle_l",
                        "mtp_angle_r",
                    ],
                    axis=1,
                )
                rmses[subject][motFile] = df_diff_squared.mean().pow(0.5)

    # compute mean df for all rmses
    list_of_rmses = [
        rmses[subject][motFile]
        for subject in rmses
        for motFile in rmses[subject]
        if rmses[subject][motFile] is not None
    ]
    concatenated_rmses = pd.concat(list_of_rmses)
    grouped = concatenated_rmses.groupby(level=0)
    mean_rmses = grouped.mean()
    for col in mean_rmses.index:
        if col.endswith("_r"):
            mean_rmses[col[:-2]] = (mean_rmses[col] + mean_rmses[col[:-2] + "_l"]) / 2
            mean_rmses = mean_rmses.drop(col)
            mean_rmses = mean_rmses.drop(col[:-2] + "_l")
    # only keep 3 significant digits

    mean_rmses = mean_rmses.round(3)

    print("mean_rmses_no_shift: ", mean_rmses)
    # export to csv
    mean_rmses.to_csv(
        os.path.join(
            args.output, args.pred_dir.split("/")[-1] + "mean_rmses_no_shift.csv"
        )
    )
