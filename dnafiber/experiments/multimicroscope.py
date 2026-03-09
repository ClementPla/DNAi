import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from dnafiber.analysis.chart import create_boxen_swarmplot, draw_protocol_arrows
from dnafiber.analysis.const import THREE_COLORS, Grader, LabelsColors
from dnafiber.analysis.dataframe import anonymise_types
from dnafiber.data.utils import mask_filepath_to_fibers
from dnafiber.experiments.common import load_dataframe
from dnafiber.experiments.deltavision.loader import load_exp11
from dnafiber.experiments.fiberQ import load_fibers_from_pred_folder
from dnafiber.experiments.fork_protection.maps import remap20_pred
from dnafiber.postprocess.fiber import Fibers
from dnafiber.experiments.deltavision.consts import TEST_IMAGES
from dnafiber.analysis.ratios import normalize_df


def default_arrows():
    analogs = [
        {
            "label": "CldU",
            "duration": "30'",
            "duration_min": 30,
            "color": LabelsColors.CLDU,
        },
        {
            "label": "IdU",
            "duration": "30'",
            "duration_min": 30,
            "color": LabelsColors.IDU,
        },
    ]
    post_treatment = {
        "label": "HU 4mM",
        "duration": "4h",
        "duration_min": 4 * 60,
        "color": LabelsColors.PRE_TREATMENT,
    }
    return analogs, post_treatment


def plot_zeiss_results(
    fiberq_pred_path: Path | str,
    gt_path: Path | str,
    dnai_pred_path: Path | str,
    arrow_ax: plt.Axes,
    ax: plt.Axes,
    tmp_file_name: str,
    normalize=False,
):
    fiberq_pred_path = Path(fiberq_pred_path)
    gt_path = Path(gt_path)
    dnai_pred_path = Path(dnai_pred_path)

    if Path(tmp_file_name).exists():
        with open(tmp_file_name, "rb") as f:
            zeiss_fiber_q = pickle.load(f)
    else:
        zeiss_fiber_q = load_fibers_from_pred_folder(fiberq_pred_path)
        with open(tmp_file_name, "wb") as f:
            pickle.dump(zeiss_fiber_q, f)

    images_names = []
    ratios = []

    for image_name, fiber_q in zeiss_fiber_q.items():
        fiber = fiber_q.only_double_copy()
        images_names += [image_name] * len(fiber.ratios)
        ratios += fiber.ratios

    graders = [Grader.OTHER] * len(ratios)
    df_zeiss = pd.DataFrame(
        {
            "Grader": graders,
            "Ratio": ratios,
            "Image Name": images_names,
        }
    )
    mapping = {
        "siBRCA1": "siBRCA1",
        "siNT": "siNT",
        "siBRCA2": "siBRCA2",
        "siTONSL-D+siBRCA1": "siTONS+b1",
        "siTONSL-D+siBRCA2": "siTONSL D",
        "siTONSL D": "siTONSL D",
    }

    df_zeiss["Type"] = df_zeiss["Image Name"].apply(
        lambda x: "-".join(x.stem.split("-")[:-1])
    )
    df_zeiss["Type"] = df_zeiss["Type"].map(mapping)
    df_gt = load_dataframe(
        dnai_pred_path,
        gt_path,
        rename_map_pred=remap20_pred,
        error_threshold=0.15,
    )
    df_gt["Type"] = df_gt["Type"].apply(lambda x: x.replace(" ", "_"))
    df_gt = df_gt[df_gt.Type.isin(df_zeiss.Type.unique())]
    # Concatenate df_gt and df_zeiss
    df_zeiss = pd.concat([df_gt, df_zeiss], ignore_index=True)

    df_zeiss = anonymise_types(df_zeiss)
    df_zeiss.Type = df_zeiss.Type.cat.remove_categories("Exp D")
    df_zeiss.Type = df_zeiss.Type.cat.remove_categories("Exp E")
    analogs, post_treatment = default_arrows()
    if normalize:
        df_zeiss = normalize_df(df_zeiss, "Exp A")
    draw_protocol_arrows(arrow_ax, analogs=analogs, post_treatment=post_treatment)
    create_boxen_swarmplot(
        df_zeiss,
        stripplot=True,
        palette=THREE_COLORS,
        rotate_xticks=0,
        color_label="Exp A" if normalize else None,
        ax=ax,
    )
    ax.set_xlabel("Zeiss", fontweight="bold")


def plot_leica_results(
    fiberq_pred_path: Path | str,
    gt_path: Path | str,
    dnai_pred_path: Path | str,
    arrow_ax: plt.Axes,
    ax: plt.Axes,
    tmp_file_name: str,
    normalize=False,
):
    fiberq_pred_path = Path(fiberq_pred_path)
    gt_path = Path(gt_path)
    dnai_pred_path = Path(dnai_pred_path)

    if Path(tmp_file_name).exists():
        with open(tmp_file_name, "rb") as f:
            fiberQ_pred = pickle.load(f)
    else:
        fiberQ_pred = load_fibers_from_pred_folder(fiberq_pred_path)
        with open(tmp_file_name, "wb") as f:
            pickle.dump(fiberQ_pred, f)

    fiberQ_ratios = []
    for image_name, fiber_q in fiberQ_pred.items():
        fiberQ_ratios += fiber_q.only_double_copy().ratios

    graders = [Grader.OTHER] * len(fiberQ_ratios)

    dnai_pred_files = dnai_pred_path.rglob("*.pkl")
    dnai_ratios = []
    for fiber_file in dnai_pred_files:
        fiber = Fibers.from_pickle(fiber_file)
        dnai_ratios += fiber.only_double_copy().filter_errors(0.5).ratios

    graders += [Grader.AI] * len(dnai_ratios)

    gt_ratios = pd.read_excel(gt_path, header=None)[2].dropna().values
    graders += [Grader.HUMAN] * len(gt_ratios)

    df = pd.DataFrame(
        {
            "Grader": graders,
            "Ratio": fiberQ_ratios + dnai_ratios + list(gt_ratios),
            "Type": ["Leica"]
            * (len(fiberQ_ratios) + len(dnai_ratios) + len(gt_ratios)),
        }
    )
    df = anonymise_types(df)
    analogs, post_treatment = default_arrows()
    draw_protocol_arrows(arrow_ax, analogs=analogs, post_treatment=post_treatment)
    create_boxen_swarmplot(
        df,
        stripplot=True,
        palette=THREE_COLORS,
        rotate_xticks=0,
        ax=ax,
    )
    ax.set_xlabel("Leica", fontweight="bold")


def plot_deltavision_exp_11_results(
    fiberq_pred_path: Path | str,
    gt_path: Path | str,
    dnai_pred_path: Path | str,
    arrow_ax: plt.Axes,
    ax: plt.Axes,
    tmp_file_name: str,
    normalize=False,
):
    fiberq_pred_path = Path(fiberq_pred_path)
    gt_path = Path(gt_path)
    dnai_pred_path = Path(dnai_pred_path)

    if Path(tmp_file_name).exists():
        with open(tmp_file_name, "rb") as f:
            deltavision_fiber_q = pickle.load(f)
    else:
        deltavision_fiber_q = load_fibers_from_pred_folder(fiberq_pred_path)
        with open(tmp_file_name, "wb") as f:
            pickle.dump(deltavision_fiber_q, f)

    images_names = []
    ratios = []

    for image_name, fiber_q in deltavision_fiber_q.items():
        image_name = image_name.parent.parent.parent.stem
        fiber = fiber_q.only_double_copy()
        images_names += [image_name] * len(fiber.ratios)
        ratios += fiber.ratios

    graders = [Grader.OTHER] * len(ratios)
    df_deltavision_fiber_q = pd.DataFrame(
        {
            "Grader": graders,
            "Ratio": 1 / np.asarray(ratios),
            "Image Name": images_names,
        }
    )

    df_dv = load_exp11(dnai_pred_path, gt_path)
    df_deltavision_fiber_q["Type"] = df_deltavision_fiber_q["Image Name"].apply(
        lambda x: "_".join(x.split("_")[0:2])
    )
    mapping = {
        "HeLa_DMSO": "DMSO",
        "HeLa_mirin": "DMSO_mirin",
        "HeLa_ML324": "ML324_DMSO",
    }

    df_deltavision_fiber_q["Type"] = df_deltavision_fiber_q["Type"].map(mapping)
    df_dv = pd.concat([df_dv, df_deltavision_fiber_q], ignore_index=True)
    df_dv = anonymise_types(df_dv)
    df_dv.Type = df_dv.Type.cat.remove_categories("Exp D")
    if normalize:
        df_dv = normalize_df(df_dv, "Exp A")
    analogs = [
        {
            "label": "IdU",
            "duration": "30'",
            "duration_min": 30,
            "color": LabelsColors.IDU,
        },
        {
            "label": "CldU",
            "duration": "30'",
            "duration_min": 30,
            "color": LabelsColors.CLDU,
        },
    ]
    draw_protocol_arrows(arrow_ax, analogs=analogs)
    create_boxen_swarmplot(
        df_dv,
        THREE_COLORS,
        stripplot=True,
        rotate_xticks=0,
        yrange=(0, 16),
        annotate=False,
        color_label="Exp A" if normalize else None,
        ax=ax,
        ylabel="IdU / CldU",
    )
    ax.set_xlabel("DeltaVision", fontweight="bold")


def plot_deltavision_incubation_time_results(
    fiberq_pred_path: Path | str,
    gt_path: Path | str,
    dnai_pred_path: Path | str,
    arrow_ax: plt.Axes,
    ax: plt.Axes,
    tmp_fiberq_file_name: str,
    tmp_gt_file_name: str,
    normalize=False,
):
    fiberq_pred_path = Path(fiberq_pred_path)
    gt_path = Path(gt_path)
    dnai_pred_path = Path(dnai_pred_path)

    if Path(tmp_fiberq_file_name).exists():
        with open(tmp_fiberq_file_name, "rb") as f:
            deltavision_fiber_q = pickle.load(f)
    else:
        deltavision_fiber_q = load_fibers_from_pred_folder(fiberq_pred_path)
        with open(tmp_fiberq_file_name, "wb") as f:
            pickle.dump(deltavision_fiber_q, f)

    images_names = []
    ratios = []
    graders = []
    for image_name, fiber_q in deltavision_fiber_q.items():
        image_name = image_name.stem
        fiber = fiber_q.only_double_copy()
        images_names += [image_name] * len(fiber.ratios)
        ratios += fiber.ratios
        graders += [Grader.OTHER] * len(fiber.ratios)

    dnai_files = dnai_pred_path.rglob("*.pkl")
    for fiber_file in dnai_files:
        fiber = Fibers.from_pickle(fiber_file)
        image_name = fiber_file.stem
        fiber = fiber.only_double_copy()
        images_names += [image_name] * len(fiber.ratios)
        ratios += fiber.ratios
        graders += [Grader.AI] * len(fiber.ratios)

    if Path(tmp_gt_file_name).exists():
        with open(tmp_gt_file_name, "rb") as f:
            gt = pickle.load(f)
    else:
        gt = {}
        gt_files = gt_path.rglob("*gt.png")
        for gt_file in gt_files:
            image_name = gt_file.parent.name
            # Find which image in the test set corresponds to this gt
            for img_test in TEST_IMAGES:
                if img_test.parent.stem == image_name:
                    break

            gt[img_test.stem] = mask_filepath_to_fibers(gt_file)
        with open(tmp_gt_file_name, "wb") as f:
            pickle.dump(gt, f)
    for gt_name, gt_fibers in gt.items():
        image_name = gt_name
        fiber = gt_fibers.only_double_copy()
        images_names += [image_name] * len(fiber.ratios)
        ratios += fiber.ratios
        graders += [Grader.HUMAN] * len(fiber.ratios)
    df_deltavision = pd.DataFrame(
        {
            "Grader": graders,
            "Ratio": np.asarray(ratios),
            "Image Name": images_names,
        }
    )
    df_deltavision["Type"] = df_deltavision["Image Name"].apply(
        lambda x: x.split("min")[0][-2:] + " min"
    )

    order = ["20 min", "30 min", "40 min", "60 min"]
    df_deltavision["Type"] = pd.Categorical(
        df_deltavision["Type"], categories=order, ordered=True
    )
    df_deltavision = df_deltavision.sort_values("Type")
    if normalize:
        df_deltavision = normalize_df(df_deltavision, "20 min")
    analogs = [
        {
            "label": "IdU",
            "duration": "30'",
            "duration_min": 20,
            "color": LabelsColors.IDU,
        },
        {
            "label": "CldU",
            "duration": "20, 30, 40, 60'",
            "duration_min": 30,
            "color": LabelsColors.CLDU,
        },
    ]

    draw_protocol_arrows(arrow_ax, analogs=analogs)
    create_boxen_swarmplot(
        df_deltavision,
        THREE_COLORS,
        stripplot=True,
        rotate_xticks=0,
        yrange=(0, 16),
        annotate=False,
        color_label="20 min" if normalize else None,
        ax=ax,
        ylabel="CldU / IdU",
    )
    ax.set_xlabel("DeltaVision - Incubation time", fontweight="bold")
