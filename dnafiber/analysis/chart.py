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
