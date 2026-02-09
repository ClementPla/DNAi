import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import f_oneway, tukey_hsd, mannwhitneyu
from scipy.stats import ttest_ind, kruskal, alexandergovern


def load_experiment_predictions(root, filter_length=None, keep_invalid=False):
    root = Path(root)
    files = list(root.rglob("*.csv"))

    # Read all dataframes and concatenate them
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        if "Image Name" in df.columns:
            df["image_name"] = df["Image Name"]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df["Length"] = df["First analog (µm)"] + df["Second analog (µm)"]
    if filter_length is not None:
        length = df["Second analog (µm)"] + df["First analog (µm)"]
        df = df[length > filter_length]
    if not keep_invalid:
        df = df[df["Valid"] == True]
    df["Type"] = df["image_name"].apply(lambda x: "-".join(x.split("-")[:-1]))
    df["Grader"] = "AI"
    df = df[df["Fiber type"] == "double"]
    return df


def load_experiment_gt(root):
    gt = pd.read_excel(root, sheet_name=None, header=None)

    # Concatenate all ground truth dataframes
    gt_dfs = []
    for sheet_name, gt_df in gt.items():
        df_gt = pd.DataFrame()
        valid_indices = gt_df[0].dropna().index
        gt_df = gt_df.loc[valid_indices]
        # Analogs: column 1, Even row: second analog, Odd row: first analog
        first_analog = gt_df[1].dropna().values[1::2]
        second_analog = gt_df[1].dropna().values[0::2]
        df_gt["Ratio"] = gt_df[2].dropna()
        df_gt["Grader"] = "Human"
        df_gt["Type"] = sheet_name
        df_gt["Length"] = first_analog + second_analog
        df_gt["Valid"] = True
        gt_dfs.append(df_gt)

    df_gt = pd.concat(gt_dfs, ignore_index=True)
    return df_gt


def load_experiment(root_pred, root_gt, filter_invalid=False, keep_n_longest=None):
    df = load_experiment_predictions(root_pred, keep_invalid=not filter_invalid)
    if filter_invalid:
        df = df[df["Valid"]]
    if keep_n_longest is not None:
        df = (
            df.sort_values(by="Length", ascending=False)
            .groupby("Type")
            .head(keep_n_longest)
        )
    if not isinstance(root_gt, list):
        root_gt = [root_gt]
    df_gts = []
    for root in root_gt:
        if isinstance(root, str):
            root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Ground truth file {root} does not exist.")
        df_gt = load_experiment_gt(root)
        df_gts.append(df_gt)
    df_gt = pd.concat(df_gts, ignore_index=True)

    df = pd.concat(
        [df[["Type", "Ratio", "Grader", "Length", "Valid"]], df_gt], ignore_index=True
    )
    df["Grader"] = pd.Categorical(
        df["Grader"], categories=["Human", "AI"], ordered=True
    )

    return df


def normalize_df(df, baseline):
    """
    Normalize the 'Ratio' column of the DataFrame.
    """
    for grader in ["Human", "AI"]:
        # Normalize the ratios by the baseline
        median_grader_baseline = df[
            (df["Type"] == baseline) & (df["Grader"] == grader)
        ]["Ratio"].median()

        # Get the mask for this grader
        mask = df["Grader"] == grader

        # Normalize using the mask - more explicit in-place modification
        df.loc[mask, "Ratio"] = df.loc[mask, "Ratio"] / median_grader_baseline

    return df


def pvalue_to_asterisk(p_value):
    """Convert p-value to asterisks."""
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


def compare_pairs(df, pairs, palette, base_offset=6, column="Ratio"):
    group_names = df["Type"].unique().tolist()

    for grader, color in zip(["Human", "AI"], palette):
        for pair in pairs:
            group1 = df[(df["Type"] == pair[0]) & (df["Grader"] == grader)][column]
            group2 = df[(df["Type"] == pair[1]) & (df["Grader"] == grader)][column]

            mannwhitney_results = mannwhitneyu(
                group1, group2, alternative="two-sided", method="exact"
            )
            pvalue = mannwhitney_results.pvalue
            index_one = group_names.index(pair[0])
            index_two = group_names.index(pair[1])
            # Add asterisks based on p-value
            asterisks = pvalue_to_asterisk(pvalue)
            if asterisks:
                # Calculate the position for the asterisks
                x1 = index_one + 0.1
                x2 = index_two - 0.1
                y = base_offset + pair[2]  # y position for the asterisks
                plt.plot(
                    [x1, x1, x2, x2],
                    [y, y + 0.05 * y, y + 0.05 * y, y],
                    lw=1.5,
                    color="black",
                )
                plt.text(
                    x=(x1 + x2) / 2 + (-0.3 if grader == "Human" else 0.3),
                    y=y,
                    s=asterisks,
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color=color,
                )


def graders_statistical_test(df, yoffset=6, column="Ratio"):
    # Perform one-way ANOVA for each grader
    for i, type_ in enumerate(df["Type"].unique()):
        group1 = df[(df["Type"] == type_) & (df["Grader"] == "Human")][column]
        group2 = df[(df["Type"] == type_) & (df["Grader"] == "AI")][column]

        if group1.empty or group2.empty:
            continue

        # Perform t-test
        t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
        try:
            p_value = f_oneway(group1, group2).pvalue
        except ValueError:
            continue

        asterisks = pvalue_to_asterisk(p_value)
        if asterisks:
            # Calculate the position for the asterisks
            x1 = i + 0.1
            x2 = i - 0.1

            plt.plot(
                [x1, x1, x2, x2],
                [yoffset, yoffset + 0.5, yoffset + 0.5, yoffset],
                lw=1.5,
                color="black",
            )
            plt.text(
                x=(x1 + x2) / 2,
                y=yoffset + 0.5,
                s=asterisks,
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
            )


def select_N_closest_to_mean(df, N=10, column="Ratio"):
    """
    Select N closest values to the mean for each Type and Grader.
    """
    selected_dfs = []
    for (type_, grader), group in df.groupby(["Type", "Grader"]):
        if grader == "Human":
            selected_dfs.append(group)
            continue
        mean = group[column].mean()
        group["Distance to mean"] = (group[column] - mean).abs()
        if N is None:
            # We chose N as the size of the Human grader group
            N = df[(df["Type"] == type_) & (df["Grader"] == "Human")].shape[0]
        closest_to_mean = group.nsmallest(N, "Distance to mean")
        selected_dfs.append(closest_to_mean)
    return pd.concat(selected_dfs, ignore_index=True)


def select_N_closest_to_median(df, N=10, column="Ratio"):
    """
    Select N closest values to the median for each Type and Grader.
    """
    selected_dfs = []
    for (type_, grader), group in df.groupby(["Type", "Grader"]):
        if grader == "Human":
            selected_dfs.append(group)
            continue
        median = group[column].median()
        group["Distance to median"] = (group[column] - median).abs()
        if N is None:
            # We chose N as the size of the Human grader group
            N = df[(df["Type"] == type_) & (df["Grader"] == "Human")].shape[0]
        closest_to_median = group.nsmallest(N, "Distance to median")
        selected_dfs.append(closest_to_median)

    df = pd.concat(selected_dfs, ignore_index=True)
    df.sort_values(["Grader"], inplace=True)

    return df


def offset_transition(df, offset=0, pixel_size=0.13, symetric=True):
    """
    Offset the 'Second analog (µm)' column by a fixed value in micrometers.
    """
    df.loc[df["Grader"] == "AI", "Second analog (µm)"] = (
        df.loc[df["Grader"] == "AI", "Second analog (µm)"] - offset * pixel_size
    )
    if symetric:
        df.loc[df["Grader"] == "AI", "First analog (µm)"] = (
            df.loc[df["Grader"] == "AI", "First analog (µm)"] + offset * pixel_size
        )
    df.loc[df["Grader"] == "AI", "Ratio"] = (
        df.loc[df["Grader"] == "AI", "Second analog (µm)"]
        / df.loc[df["Grader"] == "AI", "First analog (µm)"]
    )
    return df
