from typing import Optional
import pandas as pd
from dnafiber.analysis.const import palette
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotate_xticks, ha="right")
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle="--", alpha=0.7)


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
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotate_xticks, ha="right")

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
    **kwargs,
):
    # Always order the graders in the same way (Human first, then AI)
    df["Grader"] = pd.Categorical(
        df["Grader"], categories=["Human", "AI"], ordered=True
    )
    df.sort_values("Grader", inplace=True)
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

    if annotate:
        annotate_medians(df, ax, column)
    return plt.gcf()


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

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.3, 1.1)
    ax.axis("off")

    # Layout parameters
    y_arrows = 0.5
    arrow_height = 0.20
    thin_arrow_height = 0.12
    gap = 0.0
    analog_arrow_width = 1.8
    pt_width = 1.0
    time_y = y_arrows + arrow_height * 1.4 / 2 + 0.08

    # --- Analog arrows ---
    x_analog_start = 3.0
    x_cursor = x_analog_start

    # Total visual width for all analogs, then distribute proportionally
    total_analog_visual = 3.6  # total visual width budget for all analogs
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
            length_includes_head=True,
            head_length=0.2,
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
        x_cursor += analog_arrow_width

    x_end_analogs = x_cursor

    # --- Dotted vertical separator + post-treatment ---
    if post_treatment is not None:
        sep_x = x_end_analogs + gap / 2 + 0.15
        ax.plot(
            [sep_x, sep_x],
            [
                y_arrows - arrow_height * 1.4 / 2 - 0.15,
                y_arrows + arrow_height * 1.4 / 2 + 0.25,
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
            head_length=0.12,
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
                ha="right",
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
            head_length=0.12,
            fc=color,
            ec="none",
            alpha=0.6,
        )
        ax.add_patch(arrow)
        # Duration label to the left of the bar
        ax.text(
            x_pre_start + pre_visual_width / 2,
            y_pre + bar_height / 2 + 0.07,
            pretreatment.get("duration", ""),
            ha="right",
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
