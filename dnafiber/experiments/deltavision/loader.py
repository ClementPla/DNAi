from dnafiber.postprocess.fiber import Fibers, FiberProps
from dnafiber.analysis.const import Grader
import pandas as pd
import numpy as np


def load_exp1(root_pred, root_excel):
    gt = pd.read_excel(
        root_excel,
        header=2,
        sheet_name="totals",
    )

    gt1 = pd.DataFrame(
        {"Ratio": gt["2 div 1"], "Type": "0uM HU", "Grader": Grader.HUMAN}
    )
    gt2 = pd.DataFrame(
        {"Ratio": gt["2 div 1.1"], "Type": "50uM HU", "Grader": Grader.HUMAN}
    )
    gt3 = pd.DataFrame(
        {"Ratio": gt["2 div 1.2"], "Type": "100uM HU", "Grader": Grader.HUMAN}
    )
    gt4 = pd.DataFrame(
        {"Ratio": gt["2 div 1.3"], "Type": "200uM HU", "Grader": Grader.HUMAN}
    )

    gt_combined = pd.concat([gt1, gt2, gt3, gt4], ignore_index=True)

    folder_mapping = {
        "293_100uM_HU": "100uM HU",
        "293_200uM_HU": "200uM HU",
        "293-0-HU": "0uM HU",
        "293-50-HU": "50uM HU",
    }
    dfs = []
    for folder in folder_mapping:
        list_of_files = list((root_pred / folder).rglob("*.pkl"))
        for file in list_of_files:
            predictions = Fibers.from_pickle(file).filter_errors(0.5)
            df = predictions.to_df(pixel_size=0.0677249823915)
            df["Type"] = folder_mapping[folder]
            df["Grader"] = Grader.AI
            dfs.append(df)
    if len(dfs) > 0:
        combined_df = pd.concat(dfs, ignore_index=True)
        df = pd.concat([gt_combined, combined_df], ignore_index=True)
        return df
    else:
        raise ValueError("No prediction files found in the specified directory.")


def load_exp2(root_pred, root_excel):
    gt = pd.read_excel(
        root_excel,
        header=2,
        sheet_name="Sheet2",
    )
    gt1 = pd.DataFrame(
        {"Ratio": 1 / gt["2 div 1"], "Type": "1366_HU", "Grader": Grader.HUMAN}
    )
    gt2 = pd.DataFrame(
        {"Ratio": 1 / gt["2 div 1.1"], "Type": "1366_HU_mirin", "Grader": Grader.HUMAN}
    )
    gt3 = pd.DataFrame(
        {"Ratio": 1 / gt["2 div 1.2"], "Type": "3248_HU", "Grader": Grader.HUMAN}
    )
    gt4 = pd.DataFrame(
        {"Ratio": 1 / gt["2 div 1.3"], "Type": "3248_HU_mirin", "Grader": Grader.HUMAN}
    )

    gt_combined = pd.concat([gt1, gt2, gt3, gt4], ignore_index=True)

    folder_mapping = {
        "3284_HU_mirin": "3248_HU_mirin",
        "1366_HU": "1366_HU",
        "1366_HU_mirin": "1366_HU_mirin",
        "3248_HU": "3248_HU",
    }

    dfs = []
    for folder in folder_mapping:
        list_of_files = list((root_pred / folder).rglob("*.pkl"))
        for file in list_of_files:
            predictions = Fibers.from_pickle(file).filter_errors(0.5)
            df = predictions.to_df(pixel_size=0.0677249823915)
            df["Type"] = folder_mapping[folder]
            df["Grader"] = Grader.AI
            dfs.append(df)
    if len(dfs) > 0:
        combined_df = pd.concat(dfs, ignore_index=True)
        df = pd.concat([gt_combined, combined_df], ignore_index=True)
        return df
    else:
        raise ValueError("No prediction files found in the specified directory.")


def load_exp3(root_pred, root_excel):
    gt = pd.read_excel(
        root_excel,
        header=2,
        sheet_name="totals",
    )
    gt1 = pd.DataFrame(
        {"Ratio": gt["2 div 1"], "Type": "WM35 HU", "Grader": Grader.HUMAN}
    )
    gt2 = pd.DataFrame(
        {"Ratio": gt["2 div 1.1"], "Type": "WM278 HU", "Grader": Grader.HUMAN}
    )

    gt_combined = pd.concat([gt1, gt2], ignore_index=True)

    folder_mapping = {
        "WM35_HU": "WM35 HU",
        # "WM35_HU_RI1": "PEO4",
        "WM278_HU": "WM278 HU",
        # "WM278_HU_RI1": "PEO4",
    }

    dfs = []
    for folder in folder_mapping:
        list_of_files = list((root_pred / folder).rglob("*.pkl"))
        for file in list_of_files:
            predictions = Fibers.from_pickle(file).filter_errors(0.5)
            df = predictions.to_df(pixel_size=0.0677249823915)
            df["Type"] = folder_mapping[folder]
            df["Grader"] = Grader.AI
            dfs.append(df)
    if len(dfs) > 0:
        combined_df = pd.concat(dfs, ignore_index=True)
        df = pd.concat([gt_combined, combined_df], ignore_index=True)
        return df
    else:
        raise ValueError("No prediction files found in the specified directory.")


def load_exp4(root_pred, root_excel):
    gt = pd.read_excel(
        root_excel,
        header=2,
        sheet_name="calculations",
    )
    gt1 = pd.DataFrame(
        {"Ratio": gt["2 over 1"], "Type": "WM1314D", "Grader": Grader.HUMAN}
    )
    gt2 = pd.DataFrame(
        {"Ratio": gt["2 over 1.1"], "Type": "WM1617", "Grader": Grader.HUMAN}
    )

    gt_combined = pd.concat([gt1, gt2], ignore_index=True)

    folder_mapping = {
        "WM1341D_HU": "WM1314D",
        "WM1617_HU": "WM1617",
    }

    dfs = []
    for folder in folder_mapping:
        list_of_files = list((root_pred / folder).rglob("*.pkl"))
        for file in list_of_files:
            predictions = Fibers.from_pickle(file).filter_errors(0.5)
            df = predictions.to_df(pixel_size=0.0677249823915)
            df["Type"] = folder_mapping[folder]
            df["Grader"] = Grader.AI
            dfs.append(df)
    if len(dfs) > 0:
        combined_df = pd.concat(dfs, ignore_index=True)
        df = pd.concat([gt_combined, combined_df], ignore_index=True)
        return df
    else:
        raise ValueError("No prediction files found in the specified directory.")


def load_exp6(root_pred, root_excel):
    gt = pd.read_excel(root_excel, header=6, sheet_name="ratios")
    names = pd.read_excel(root_excel, header=5, sheet_name="ratios")
    names = [c for c in names.columns if "Unnamed" not in c]
    gts = []
    for i, name in enumerate(names):
        suffix = f".{i}" if i > 0 else ""
        gti = pd.DataFrame(
            {"Ratio": gt["1 div 2" + suffix], "Type": name, "Grader": "Human"}
        )
        gts.append(gti)

    gt_combined = pd.concat(gts, ignore_index=True)
    folder_mapping = {
        "20170828_HeLa100MLHU_8": "100 ML324+HU",
        "20170828_HeLa100EtOHHU_7": "100 EtOH+HU",
        "20170829_100ML_noHU_6": "100 ML324 no HU",
        "20170829_100EtOH_5": "100 EtOH no HU",
        "20170829_HUonly_9": "HU only",
        "20170831_50ML_HU_4": "50 ML HU",
        "20170831_50EtOH_HU_3": "50 EtOH HU",
        "20170910_50ML_noHU_2": "50 ML no HU",
        "20170910_50EtOH_noHU_1": "50 EtOH no HU",
    }

    dfs = []
    for folder in folder_mapping:
        list_of_files = list((root_pred / folder).rglob("*.pkl"))
        for file in list_of_files:
            predictions = Fibers.from_pickle(file).filter_errors(0.5)
            df = predictions.to_df(pixel_size=0.0677249823915)
            df["Type"] = folder_mapping[folder]
            df["Grader"] = Grader.AI
            dfs.append(df)
    if len(dfs) > 0:
        combined_df = pd.concat(dfs, ignore_index=True)
        df = pd.concat([gt_combined, combined_df], ignore_index=True)
        return df
    else:
        raise ValueError("No prediction files found in the specified directory.")


def load_exp7(root_pred, root_excel):
    # Load human grading
    gt = pd.read_excel(root_excel, header=[2, 3], sheet_name="ratios")
    conditions = np.unique([col[0] for col in gt.columns[1:]])
    gts = []
    for condition in conditions:
        gti = pd.DataFrame(
            {
                "Ratio": gt[(condition, "red/green")],
                "Type": condition,
                "Grader": "Human",
            }
        )
        gts.append(gti)
    gt_combined = pd.concat(gts, ignore_index=True)

    # Load AI predictions
    dfs = []
    for dir in root_pred.iterdir():
        if not dir.is_dir():
            continue
        list_of_files = list(dir.rglob("*.pkl"))
        for file in list_of_files:
            predictions = Fibers.from_pickle(file).filter_errors(0.5)
            df = predictions.to_df(pixel_size=0.0677249823915)
            df["Type"] = dir.stem.replace("_", " ").replace("20170919 ", "")
            df["Grader"] = Grader.AI
            dfs.append(df)

    if len(dfs) > 0:
        combined_df = pd.concat(dfs, ignore_index=True)
        df = pd.concat([gt_combined, combined_df], ignore_index=True)
        return df
    else:
        raise ValueError("No prediction files found in the specified directory.")


def load_exp8(root_pred, root_excel):
    # Load human grading
    gt = pd.read_excel(root_excel, header=[2, 3], sheet_name="ratios")
    conditions = np.unique([col[0] for col in gt.columns[1:]])
    gts = []
    for condition in conditions:
        gti = pd.DataFrame(
            {
                "Ratio": gt[(condition, "ratio")],
                "Type": condition,
                "Grader": "Human",
            }
        )
        gts.append(gti)
    gt_combined = pd.concat(gts, ignore_index=True)

    # Load AI predictions
    dfs = []
    for dir in root_pred.iterdir():
        if not dir.is_dir():
            continue
        list_of_files = list(dir.rglob("*.pkl"))
        for file in list_of_files:
            predictions = Fibers.from_pickle(file).filter_errors(0.5)
            df = predictions.to_df(pixel_size=0.0677249823915)
            df["Type"] = dir.stem.replace("_", " ").replace("20170919 ", "")
            df["Grader"] = Grader.AI
            dfs.append(df)

    if len(dfs) > 0:
        combined_df = pd.concat(dfs, ignore_index=True)
        df = pd.concat([gt_combined, combined_df], ignore_index=True)
    else:
        raise ValueError("No prediction files found in the specified directory.")

    # Rename conditions to standardized names
    df.replace(
        {
            "Type": {
                "0 mM 2HG 0 µM HU": "HeLa 0HU 0HG",
                "0 mM 2HG 200 µM HU": "HeLa 200HU 02HG",
                "4 mM 2HG 0 µM HU": "HeLa 0HU 4mM2HG",
                "4 mM 2HG 200 µM HU": "HeLa 200HU 4mM2HG",
            }
        },
        inplace=True,
    )

    return df


def load_exp10(root_pred, root_excel):
    # Load human grading
    gt = pd.read_excel(root_excel, header=[3, 4], sheet_name="Sheet2")
    conditions = np.unique([col[0] for col in gt.columns[1:]])
    gts = []
    for condition in conditions:
        gti = pd.DataFrame(
            {
                "Ratio": 1 / gt[(condition, "green/red")],
                "Type": condition,
                "Grader": "Human",
            }
        )
        gts.append(gti)
    gt_combined = pd.concat(gts, ignore_index=True)

    # Load AI predictions
    dfs = []
    for dir in root_pred.iterdir():
        if not dir.is_dir():
            continue
        list_of_files = list(dir.rglob("*.pkl"))
        for file in list_of_files:
            predictions = Fibers.from_pickle(file).filter_errors(0.5)
            df = predictions.to_df(pixel_size=0.0677249823915)
            df["Type"] = dir.stem.replace("_A", "")
            df["Grader"] = Grader.AI
            dfs.append(df)

    if len(dfs) > 0:
        combined_df = pd.concat(dfs, ignore_index=True)
        df = pd.concat([gt_combined, combined_df], ignore_index=True)
    else:
        raise ValueError("No prediction files found in the specified directory.")

    # Rename conditions to standardized names
    df.replace(
        {
            "Type": {
                "DMSO1": "DMSO",
                "2HG1": "2HG",
            }
        },
        inplace=True,
    )

    return df


def load_exp11(root_pred, root_excel):
    # Load human grading
    gt = pd.read_excel(root_excel, header=[1, 2], sheet_name="tables")
    conditions = np.unique([col[0] for col in gt.columns[1:]])
    gts = []
    for condition in conditions:
        gti = pd.DataFrame(
            {
                "Ratio": 1 / gt[(condition, "green/red")],
                "Type": condition,
                "Grader": "Human",
            }
        )
        gts.append(gti)
    gt_combined = pd.concat(gts, ignore_index=True)

    # Load AI predictions
    dfs = []
    for dir in root_pred.iterdir():
        if not dir.is_dir():
            continue
        list_of_files = list(dir.rglob("*.pkl"))
        for file in list_of_files:
            predictions = Fibers.from_pickle(file).filter_errors(0.5)
            df = predictions.to_df(pixel_size=0.0677249823915)
            df["Type"] = (
                dir.stem.replace("HeLa_", "")
                .replace("DMSO_1", "DMSO")
                .replace("mirin", "DMSO_mirin")
            )
            df["Grader"] = Grader.AI
            dfs.append(df)

    if len(dfs) > 0:
        combined_df = pd.concat(dfs, ignore_index=True)
        df = pd.concat([gt_combined, combined_df], ignore_index=True)
    else:
        raise ValueError("No prediction files found in the specified directory.")

    # Rename conditions to standardized names
    df.replace(
        {
            "Type": {
                "ML324_DMSO_mirin": "ML324_mirin",
            }
        },
        inplace=True,
    )

    return df
