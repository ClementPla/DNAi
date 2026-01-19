from pathlib import Path
from typing import Optional
import pandas as pd
import pickle
from dnafiber.postprocess.fiber import Fibers
import numpy as np


def load_spreadsheet(root_excel: Path | str) -> pd.DataFrame:
    excel_files = pd.ExcelFile(root_excel)
    df_gts = pd.DataFrame()
    for sheet_name in excel_files.sheet_names:
        df_sheet = excel_files.parse(sheet_name, header=None).iloc[:, :3]
        if 1 not in df_sheet.columns:
            continue
        df_gt = pd.DataFrame()

        # First, let's check if there is valid data in column 2
        # Look at non-na values in column 2
        if 2 not in df_sheet.columns or df_sheet[2].dropna().empty:
            # We are only having Second Analog lengths
            df_gt["Second analog (µm)"] = df_sheet[1].dropna().reset_index(drop=True)
            df_gt["First analog (µm)"] = 0
            df_gt["Length"] = df_gt["Second analog (µm)"]
            df_gt["Type"] = sheet_name
            df_gt["Fiber type"] = "double"
            df_gt["Ratio"] = np.nan
            df_gts = pd.concat([df_gts, df_gt], ignore_index=True)
            continue

        try:
            df_gt["Ratio"] = df_sheet[2].astype(float).dropna().reset_index(drop=True)
        except KeyError:
            continue
        # even row is SecondAnalog, odd is FirstAnalog
        df_gt["First analog (µm)"] = df_sheet.index.map(
            lambda x: df_sheet.iloc[x, 1] if x % 2 == 1 else np.nan
        ).dropna()
        df_gt["Second analog (µm)"] = df_sheet.index.map(
            lambda x: df_sheet.iloc[x, 1] if x % 2 == 0 else np.nan
        ).dropna()

        df_gt["Length"] = df_gt["First analog (µm)"] + df_gt["Second analog (µm)"]
        df_gt["Type"] = sheet_name

        df_gt["Fiber type"] = "double"

        df_gts = pd.concat([df_gts, df_gt], ignore_index=True)
    return df_gts


def default_map_condition_name(name: str):
    return name.replace("_", " ").replace("-", " ")


def load_dataframe(
    root_pred: Path | str,
    root_gt: Path | str,
    rename_map_gt: Optional[callable] = None,
    rename_map: Optional[callable] = default_map_condition_name,
) -> pd.DataFrame:
    root_pred = Path(root_pred)
    root_gt = Path(root_gt)

    pred_files = list(Path(root_pred).rglob("*.pkl"))
    df_preds = None
    for pred_file in pred_files:
        with open(pred_file, "rb") as f:
            fiber_preds: Fibers = pickle.load(f)
        if len(fiber_preds) == 0:
            continue
        img_name = pred_file.stem
        df_pred = fiber_preds.only_double_copy().to_df()
        df_pred["Type"] = img_name.split("-0")[0]

        if df_preds is None:
            df_preds = df_pred
        else:
            df_preds = pd.concat([df_preds, df_pred], ignore_index=True)
        if df_preds is None:
            raise ValueError("No predictions found in the specified directory.")
    # Read the spreadsheets with pandas
    df_gts = load_spreadsheet(root_gt)
    if rename_map_gt is not None:
        df_gts["Type"] = df_gts["Type"].apply(rename_map_gt)
    df_gts["Grader"] = "Human"
    df_preds["Grader"] = "AI"
    df_preds["Type"] = df_preds["Type"]
    df_preds["Length"] = df_preds["First analog (µm)"] + df_preds["Second analog (µm)"]

    df_all = pd.concat([df_gts, df_preds], ignore_index=True)

    if rename_map is not None:
        df_all["Type"] = df_all["Type"].apply(rename_map)

    df_all.drop(["Fiber ID", "Valid"], axis=1, inplace=True, errors="ignore")
    return df_all
