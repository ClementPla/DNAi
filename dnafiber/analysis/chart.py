from typing import Optional
import pandas as pd
from dnafiber.analysis.const import palette
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr, spearmanr
from dnafiber.analysis.const import Grader
from matplotlib.patches import Patch

import warnings

# Filter userwarning from matplotlib about log scale and non-positive values
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message="set_ticklabels.*FixedLocator")
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")


def get_color_association(df):
    """
    Get the color association for each image in the dataframe.
    """
    unique_name = df["image_name"].unique()
    color_association = {i: p for (i, p) in zip(unique_name, palette)}
    return color_association


def plot_ratio(df, color_association=None, only_bilateral=True):
    df = df[["ratio", "image_name", "fiber_type"]].copy()

    df["Image"] = df["image_name"]
    df["Fiber Type"] = df["fiber_type"]
    df["Ratio"] = df["ratio"]
    if only_bilateral:
        df = df[df["Fiber Type"] == "double"]

    df = df.sort_values(
        by=["Image", "Fiber Type"],
        ascending=[True, True],
    )

    # Order the dataframe by the average ratio of each image
    image_order = (
        df.groupby("Image")["Ratio"].median().sort_values(ascending=True).index
    )
    df["Image"] = pd.Categorical(df["Image"], categories=image_order, ordered=True)
    df.sort_values("Image", inplace=True)
    if color_association is None:
        color_association = get_color_association(df)
    unique_name = df["image_name"].unique()
    color_association = {i: p for (i, p) in zip(unique_name, palette)}

    this_palette = [color_association[i] for i in unique_name]
    fig = px.violin(
        df,
        y="Ratio",
        x="Image",
        color="Image",
        color_discrete_sequence=this_palette,
        box=True,  # draw box plot inside the violin
        points="all",  # can be 'outliers', or False
    )

    # Make the fig taller

    fig.update_layout(
        height=500,
        width=1000,
        title="Ratio of green to red",
        yaxis_title="Ratio",
        xaxis_title="Image",
        legend_title="Image",
    )
    return fig


def create_boxen_plot(
    df,
    palette,
    yrange=(0.125, 32),
    column="Ratio",
    log_scale=True,
    rotate_xticks=45,
    ax=None,
    show_counts=False,
    **kwargs,
):
    sns.boxplot(
        data=df,
        x="Type",
        y=column,
        hue="Grader",
        palette=palette,
        linewidth=0.75,
        ax=ax,
        **kwargs,
    )
    if ax is None:
        ax = plt.gca()
    if log_scale:
        ax.set_yscale("log")
        ax.set_yticks([0.125, 0.25, 0.5, 1, 2, 4, 8])
        ax.set_yticklabels([0.125, 0.25, 0.5, 1, 2, 4, 8])
    ax.minorticks_off()
    ax.set_ylim(*yrange)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotate_xticks, ha="center")
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    if show_counts:
        types = (
            df["Type"].cat.categories
            if hasattr(df["Type"], "cat")
            else df["Type"].unique()
        )
        graders = (
            df["Grader"].cat.categories
            if hasattr(df["Grader"], "cat")
            else df["Grader"].unique()
        )
        n_graders = len(graders)

        for i, typ in enumerate(types):
            for j, grader in enumerate(graders):
                count = len(df[(df["Type"] == typ) & (df["Grader"] == grader)])
                # Position each grader's label above its box
                x_pos = i + (j - (n_graders - 1) / 2) * (0.8 / n_graders)
                ax.text(
                    x_pos,
                    4,
                    f"N={count}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color=palette[j] if isinstance(palette, list) else "black",
                )


def create_swarm_plot(
    df,
    palette,
    yrange=(0.125, 32),
    column="Ratio",
    log_scale=True,
    include_median=True,
    rotate_xticks=45,
    stripplot=False,
    ax=None,
    **kwargs,
):
    if stripplot:
        sns.stripplot(
            data=df,
            x="Type",
            y=column,
            hue="Grader",
            palette=palette,
            dodge=True,
            ax=ax,
            **kwargs,
        )
    else:
        sns.swarmplot(
            data=df, x="Type", y=column, hue="Grader", palette=palette, ax=ax, **kwargs
        )
    if ax is None:
        ax = plt.gca()
    for c in ax.collections:
        c.set_zorder(1)  # Set the zorder to 1 for all points
    if include_median:
        # Show the median as a horizontal line
        for j, grader in enumerate(df["Grader"].unique()):
            median_values = df[df["Grader"] == grader].groupby("Type")["Ratio"].median()
            offset = (
                0.2 if j == 1 else -0.2
            )  # Offset for the median line based on grader
            for i, median in enumerate(median_values):
                ax.hlines(
                    median,
                    i + offset - 0.1,
                    i + offset + 0.1,
                    colors="red",
                    linestyles="dashed",
                    lw=1.5,
                )
                # The hlines should be over the points
                ax.collections[-1].set_zorder(2)

    if log_scale:
        ax.set_yscale("log")
        ax.set_yticks([0.125, 0.25, 0.5, 1, 2, 4, 8])
        ax.set_yticklabels([0.125, 0.25, 0.5, 1, 2, 4, 8])
    ax.minorticks_off()
    ax.set_ylim(*yrange)
    # Set anchor of xticks to right
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotate_xticks, ha="center")

    ax.set_xlabel("")
    ax.grid(axis="y", linestyle="--", alpha=0.7)


def create_boxen_swarmplot(
    df,
    palette,
    yrange=(0.125, 32),
    column="Ratio",
    color_label: Optional[str] = None,
    log_scale=True,
    stripplot=False,
    rotate_xticks=45,
    size=3,
    ax=None,
    annotate=False,
    ylabel=None,
    show_counts=False,
    **kwargs,
):
    # Always order the graders in the same way (Human first, then AI)
    if len(df["Grader"].unique()) == 2:
        df["Grader"] = pd.Categorical(
            df["Grader"], categories=[Grader.HUMAN, Grader.AI], ordered=True
        )
    elif len(df["Grader"].unique()) == 3:
        df["Grader"] = pd.Categorical(
            df["Grader"],
            categories=[Grader.HUMAN, Grader.AI, Grader.OTHER],
            ordered=True,
        )
    df.sort_values(["Grader"], inplace=True)
    create_boxen_plot(
        df,
        palette=palette,
        rotate_xticks=rotate_xticks,
        yrange=yrange,
        showfliers=False,
        log_scale=log_scale,
        notch=True,
        boxprops=dict(alpha=0.5),
        column=column,
        ax=ax,
        show_counts=show_counts,
        **kwargs,
    )
    create_swarm_plot(
        df,
        include_median=False,
        palette=palette,
        yrange=yrange,
        stripplot=stripplot,
        alpha=0.8,
        column=column,
        log_scale=log_scale,
        rotate_xticks=rotate_xticks,
        size=size,
        # jitter=0.1,
        #
        ax=ax,
        legend=False,
        **kwargs,
    )
    if ax is None:
        ax = plt.gca()
    if color_label is not None:
        for label in ax.get_xticklabels():
            if label.get_text() == color_label:
                label.set_color("red")
                label.set_fontweight("bold")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if annotate:
        annotate_medians(df, ax, column)
    return plt.gcf()


def highlight_disagreements(ax, df, disagreement_types, color="#FFCCCC", alpha=0.3):
    """Add background shading behind disagreement types on a categorical axis."""
    # Get the x-tick labels and their positions
    types = [t.get_text() for t in ax.get_xticklabels()]

    for typ in disagreement_types:
        if typ in types:
            idx = types.index(typ)
            ax.axvspan(idx - 0.5, idx + 0.5, color=color, alpha=alpha, zorder=0)

    # Add a single legend entry for the shading

    existing_handles, existing_labels = ax.get_legend_handles_labels()
    shading_patch = Patch(facecolor=color, alpha=alpha, label="Grader disagreement")
    ax.legend(
        handles=existing_handles + [shading_patch],
        labels=existing_labels + ["Grader disagreement"],
    )


def add_disagreement_brackets(
    ax,
    df_results,
    disagreement_types,
    type_order,
    reference_type,
    grader_colors=None,
    y_start=None,
    bracket_spacing_factor=1.12,
    bracket_height_factor=1.03,
    annotate=True,
):
    if y_start is None:
        y_start = ax.get_ylim()[1] * 1.1

    ref_idx = type_order.index(reference_type)
    graders = [
        c.replace("delta_", "") for c in df_results.columns if c.startswith("delta_")
    ]
    graders.sort(reverse=True)

    if grader_colors is None:
        grader_colors = {g: "dimgray" for g in graders}

    sorted_types = sorted(
        disagreement_types, key=lambda t: abs(type_order.index(t) - ref_idx)
    )

    for i, typ in enumerate(sorted_types):
        row = df_results[df_results["Type"] == typ].iloc[0]
        typ_idx = type_order.index(typ)

        left = min(ref_idx, typ_idx) + 0.1
        right = max(ref_idx, typ_idx) - 0.1

        y_base = y_start * (bracket_spacing_factor**i)
        y_top = y_base * bracket_height_factor

        ax.plot(
            [left, left, right, right],
            [y_base, y_top, y_top, y_base],
            color="dimgray",
            linewidth=1.0,
            clip_on=False,
        )

        if annotate:
            mid_x = (left + right) / 2
            # Place each grader's symbol side by side
            n_graders = len(graders)
            total_width = 0.75 * (n_graders - 1)
            x_positions = np.linspace(
                mid_x - total_width / 2, mid_x + total_width / 2, n_graders
            )

            for x_pos, grader in zip(x_positions, graders):
                ez = row[f"excludes_zero_{grader}"]
                symbol = "✱" if ez else "n.s."
                ax.annotate(
                    symbol,
                    xy=(x_pos, y_top),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold" if ez else "normal",
                    color=grader_colors.get(grader, "dimgray"),
                    clip_on=False,
                )

    top_y = (
        y_start * (bracket_spacing_factor ** len(sorted_types)) * bracket_spacing_factor
    )
    current_ylim = ax.get_ylim()
    ax.set_ylim(top=max(current_ylim[1], top_y))


def annotate_medians(df, ax, column):
    # Calculate medians for each Type-Grader combination
    medians = df.groupby(["Type", "Grader"])[column].median().unstack()

    # Get types in the SAME ORDER as they appear in the plot
    types = [label.get_text() for label in ax.get_xticklabels()]
    graders = df["Grader"].cat.categories  # Use categorical order

    # Calculate positions for dodged points
    n_graders = len(graders)
    offsets = np.linspace(-0.2, 0.2, n_graders)

    for i, type_val in enumerate(types):
        # Get median values for both graders
        med1 = medians.loc[type_val, graders[0]]
        med2 = medians.loc[type_val, graders[1]]

        # Calculate x positions
        x1 = i + offsets[0]
        x2 = i + offsets[1]

        # Calculate delta
        delta = med2 - med1

        # Draw line at a fixed height above the plot
        line_y = max(med1, med2) + 3.5
        ax.plot(
            [x1, x2],
            [line_y, line_y],
            color="black",
            linewidth=1.0,
            linestyle="-",
            alpha=0.7,
            zorder=10,
        )

        # Add text annotation
        mid_x = (x1 + x2) / 2
        ax.text(
            mid_x,
            line_y + 0.5,
            f"δ={delta:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8
            ),
        )


def create_violin_plot(df, palette, ax=None):
    len_graders = len(df["Grader"].unique())
    if len_graders != 2:
        sns.violinplot(
            data=df,
            x="Type",
            y="Ratio",
            split=False,
            inner="box",
            palette=palette,
            saturation=1.0,
            gap=0.1,
            linewidth=0.75,
            ax=ax,
        )
    else:
        sns.violinplot(
            data=df,
            x="Type",
            y="Ratio",
            hue="Grader",
            split=True,
            inner="box",
            palette=palette,
            saturation=1.0,
            gap=0.01,
            linewidth=0.75,
            ax=ax,
        )

    # The yscale should be log
    # The y ticks should be in powers of 2 (0.125, 0.25, 0.5, 1, 2, 4, 8)
    ax.set_yscale("log")
    ax.set_yticks([0.125, 0.25, 0.5, 1, 2, 4, 8])
    ax.set_yticklabels([0.125, 0.25, 0.5, 1, 2, 4, 8])
    # The y range should be from 0.125 to 8
    # Remove minor ticks
    ax.minorticks_off()
    ax.set_ylim(0.125, 10)

    # Change the background color of the plot to white
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    # Remove x label
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle="--", alpha=0.7)


def draw_protocol_arrows(
    ax,
    pretreatment: dict | None = None,
    analogs: list[dict] = None,
    post_treatment: dict | None = None,
):
    """
    Draw a protocol diagram with incubation arrows on a given matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    pretreatment : dict or None
        Keys: "label" (str), "duration_min" (float, total duration in minutes),
        "color" (str), "duration" (str, display label e.g. "3h").
        Drawn below the analog arrows. Its arrow ENDS at the end of the last analog
        and extends left proportionally to its duration.
    analogs : list of dict
        Each dict has keys: "label" (str), "duration" (str, display label),
        "duration_min" (float, duration in minutes), "color" (str).
    post_treatment : dict or None
        Keys: "label" (str), "duration" (str), "color" (str),
        optionally "prefix" (str, e.g. "+/-").
    """
    import matplotlib.patches as mpatches

    if analogs is None:
        analogs = []

    ax.set_xticks([])  # Remove x-ticks
    ax.set_yticks([])  # Remove y-ticks
    ax.set_xlabel("")  # Ensure no label
    ax.spines["bottom"].set_visible(False)  # Optional: hide the bottom line

    # Layout parameters
    y_arrows = 0.5
    arrow_height = 0.20
    thin_arrow_height = 0.15
    gap = 0.0
    analog_arrow_width = 1.8
    pt_width = 1.0
    time_y = y_arrows + arrow_height / 2

    margin = 0.5
    x_start = margin

    # Budget the full width
    available = 10.0 - 2 * margin
    if post_treatment is not None:
        pt_width = available * 0.15  # 15% for post-treatment
        sep_gap = 0.3
        total_analog_visual = available - pt_width - sep_gap
    else:
        total_analog_visual = available

    x_analog_start = x_start
    x_cursor = x_analog_start

    # Total visual width for all analogs, then distribute proportionally
    analog_total_min = sum(a.get("duration_min", 30) for a in analogs)

    for analog in analogs:
        dur = analog.get("duration_min", 30)
        analog_arrow_width = total_analog_visual * (dur / analog_total_min)
        color = analog.get("color", "red")
        arrow = mpatches.FancyArrow(
            x_cursor,
            y_arrows,
            analog_arrow_width,
            0,
            width=arrow_height,
            head_width=arrow_height * 1.4,
            head_length=min(analog_arrow_width * 0.2, 1),
            length_includes_head=True,
            fc=color,
            ec="none",
        )
        ax.add_patch(arrow)
        ax.text(
            x_cursor + analog_arrow_width / 2,
            time_y,
            analog.get("duration", ""),
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax.text(
            x_cursor + analog_arrow_width / 2 - 0.1,
            y_arrows,
            analog.get("label", ""),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )
        event = analog.get("event_before", None)
        if event is not None:
            arrow_start_y = y_arrows - arrow_height * 1.4 / 2 - 0.15
            ax.annotate(
                "",
                xy=(x_cursor, y_arrows),  # head at center of analog
                xytext=(x_cursor, arrow_start_y),  # starts below
                arrowprops=dict(
                    arrowstyle="->",
                    color=event.get("color", "black"),
                    lw=1.5,
                ),
            )
            ax.text(
                x_cursor,
                arrow_start_y - 0.025,
                event.get("label", ""),
                ha="center",
                va="top",
                fontsize=8,
                fontweight="bold",
                color=event.get("color", "black"),
            )
        x_cursor += analog_arrow_width

    x_end_analogs = x_cursor

    # --- Dotted vertical separator + post-treatment ---
    if post_treatment is not None:
        sep_x = x_end_analogs + gap / 2 + 0.15
        ax.plot(
            [sep_x, sep_x],
            [
                y_arrows - arrow_height / 2 - 0.15,
                y_arrows + arrow_height / 2 + 0.25,
            ],
            linestyle=":",
            color="black",
            linewidth=1.2,
        )
        x_pt_start = sep_x + 0.15
        color = post_treatment.get("color", "blue")
        arrow = mpatches.FancyArrow(
            x_pt_start,
            y_arrows,
            pt_width,
            0,
            width=thin_arrow_height,
            head_width=thin_arrow_height * 1.4,
            length_includes_head=True,
            head_length=min(pt_width * 0.2, 1),
            fc=color,
            ec="none",
        )
        ax.add_patch(arrow)
        prefix = post_treatment.get("prefix", "")
        if prefix:
            ax.text(
                x_pt_start - 0.05,
                y_arrows,
                prefix,
                ha="center",
                va="center",
                fontsize=9,
            )
        ax.text(
            x_pt_start + pt_width / 2,
            time_y,
            post_treatment.get("duration", ""),
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax.text(
            x_pt_start + pt_width / 2,
            y_arrows - thin_arrow_height / 2 - 0.08,
            post_treatment.get("label", ""),
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
        )

    # --- Pretreatment bar (below analogs, ending at x_end_analogs) ---
    if pretreatment is not None:
        y_pre = y_arrows - arrow_height * 1.4 / 2 - 0.22
        bar_height = 0.10

        # Compute how much wider the pretreatment is vs the analog section
        analog_total_min = sum(a.get("duration_min", 30) for a in analogs)
        pre_total_min = pretreatment.get("duration_min", analog_total_min)
        analogs_visual_width = x_end_analogs - x_analog_start
        # Scale: 1 minute = analogs_visual_width / analog_total_min in visual units
        pre_visual_width = analogs_visual_width * (pre_total_min / analog_total_min)

        # Pretreatment ends at x_end_analogs, starts earlier
        x_pre_start = x_end_analogs - pre_visual_width
        color = pretreatment.get("color", "grey")

        arrow = mpatches.FancyArrow(
            x_pre_start,
            y_pre,
            pre_visual_width,
            0,
            width=bar_height,
            length_includes_head=True,
            head_width=bar_height * 1.3,
            head_length=min(pre_visual_width * 0.2, 1),
            fc=color,
            ec="none",
            alpha=0.6,
        )
        ax.add_patch(arrow)
        # Duration label to the left of the bar
        ax.text(
            x_pre_start + pre_visual_width / 2,
            y_pre + bar_height / 2 + 0.05,
            pretreatment.get("duration", ""),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
        # Label below the bar
        ax.text(
            x_pre_start + pre_visual_width / 2,
            y_pre - bar_height / 2 - 0.07,
            pretreatment.get("label", ""),
            ha="center",
            va="top",
            fontsize=9,
        )
    # 1. Horizontal Scaling: Find the start and end of all drawn objects
    left_bound = x_start
    if pretreatment is not None:
        left_bound = min(left_bound, x_pre_start)

    right_bound = x_end_analogs
    if post_treatment is not None:
        right_bound = x_pt_start + pt_width

    # 2. Vertical Scaling: Find the top and bottom
    # Analog labels are at 'time_y', Pretreatment labels are below 'y_pre'
    top_bound = time_y + 0.15
    bottom_bound = (y_pre - 0.25) if pretreatment is not None else (y_arrows - 0.4)

    # 3. Apply the limits with a tiny bit of padding
    ax.set_xlim(left_bound - 0.2, right_bound + 0.5)
    ax.set_ylim(bottom_bound, top_bound)


def regression_plot(df, ax=None, color="blue", add_stats=True):
    df_grouped = df.groupby(["Type", "Grader"]).median(numeric_only=True)
    df_pivot = df_grouped["Ratio"].unstack("Grader").reset_index()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    sns.regplot(
        x=Grader.HUMAN,
        y=Grader.AI,
        data=df_pivot,
        ax=ax,
        line_kws={"color": "black"},
        scatter_kws={"color": color},
    )
    ax.plot([0, 1.8], [0, 1.8], color="red", linestyle="--")

    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_xlim(0.65, 1.85)
    ax.set_ylim(0.65, 1.85)
    ax.set_xlabel(
        f"{Grader.HUMAN} median ratio (N={len(df[df['Grader'] == Grader.HUMAN])} fibers)"
    )
    ax.set_ylabel(
        f"{Grader.AI} median ratio (N={len(df[df['Grader'] == Grader.AI])} fibers)"
    )
    if add_stats:
        # Calculate Pearson and Spearman correlations
        pearson_corr, pearson_p = pearsonr(df_pivot[Grader.HUMAN], df_pivot[Grader.AI])
        spearman_corr, spearman_p = spearmanr(
            df_pivot[Grader.HUMAN], df_pivot[Grader.AI]
        )
        # Annotate the plot with correlation coefficients and p-values
        ax.text(
            0.05,
            0.95,
            f"Pearson r={pearson_corr:.2f} (p={pearson_p:.3g})\nSpearman ρ={spearman_corr:.2f} (p={spearman_p:.3g})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8
            ),
        )
    return ax
