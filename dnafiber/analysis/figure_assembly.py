import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any


def assemble_figures(
    image_paths: Optional[List[Union[str, Path]]] = None,
    output_path: str = "assembled_figure.png",
    layout: Optional[Tuple[int, int]] = None,
    panels: Optional[List[Dict[str, Any]]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    label_position: str = "top-left",
    label_offset: Tuple[float, float] = (0.05, 0.95),
    label_fontsize: int = 16,
    label_fontweight: str = "bold",
    label_color: str = "black",
    label_style: str = "A",  # "A", "a", "(A)", or "A)"
    spacing: float = 0.02,
    background_color: str = "white",
    a4_width_constraint: bool = True,
    max_width_inches: float = 8.27,  # A4 width in inches
) -> None:
    """
    Assemble multiple PNG images into a single figure with panel labels.

    Two modes available:
    1. Simple mode: Provide image_paths and optional layout for uniform grid
    2. Advanced mode: Provide panels list for non-uniform grid with spanning

    Parameters
    ----------
    image_paths : Optional[List[Union[str, Path]]]
        List of paths to PNG images to assemble (for simple uniform grid mode)
    output_path : str
        Path where the assembled figure will be saved
    layout : Optional[Tuple[int, int]]
        (rows, cols) for the layout. If None, automatically determined
    panels : Optional[List[Dict[str, Any]]]
        Advanced layout specification. Each dict must contain:
        - 'image': path to image file
        - 'row': starting row (0-indexed)
        - 'col': starting column (0-indexed)
        - 'rowspan': number of rows to span (default: 1)
        - 'colspan': number of columns to span (default: 1)
        - 'label': optional custom label (overrides automatic labeling)
    figsize : Optional[Tuple[float, float]]
        Figure size in inches. If None, automatically determined
    dpi : int
        Resolution of output figure
    label_position : str
        Position of labels: 'top-left', 'top-right', 'bottom-left', 'bottom-right'
    label_offset : Tuple[float, float]
        (x, y) offset for label position in axes coordinates (0-1)
    label_fontsize : int
        Font size for panel labels
    label_fontweight : str
        Font weight for labels ('normal', 'bold', etc.)
    label_color : str
        Color of the panel labels (any matplotlib color: 'black', 'red', '#FF0000', etc.)
    label_style : str
        Label style: "A" (A, B, C), "a" (a, b, c), "(A)" ((A), (B), (C)), or "A)" (A), B), C))
    spacing : float
        Spacing between subplots as fraction of figure size
    background_color : str
        Background color of the figure
    a4_width_constraint : bool
        If True, constrains figure width to A4 paper width
    max_width_inches : float
        Maximum width in inches (A4 width = 8.27 inches)
    """

    # Determine mode: simple (uniform grid) or advanced (custom panel layout)
    if panels is not None:
        # Advanced mode with custom layout
        return _assemble_advanced_layout(
            panels=panels,
            output_path=output_path,
            figsize=figsize,
            dpi=dpi,
            label_position=label_position,
            label_offset=label_offset,
            label_fontsize=label_fontsize,
            label_fontweight=label_fontweight,
            label_color=label_color,
            label_style=label_style,
            spacing=spacing,
            background_color=background_color,
            a4_width_constraint=a4_width_constraint,
            max_width_inches=max_width_inches,
        )
    elif image_paths is not None:
        # Simple mode with uniform grid
        return _assemble_simple_layout(
            image_paths=image_paths,
            output_path=output_path,
            layout=layout,
            figsize=figsize,
            dpi=dpi,
            label_position=label_position,
            label_offset=label_offset,
            label_fontsize=label_fontsize,
            label_fontweight=label_fontweight,
            label_color=label_color,
            label_style=label_style,
            spacing=spacing,
            background_color=background_color,
            a4_width_constraint=a4_width_constraint,
            max_width_inches=max_width_inches,
        )
    else:
        raise ValueError("Must provide either 'image_paths' or 'panels'")


def _assemble_simple_layout(
    image_paths: List[Union[str, Path]],
    output_path: str,
    layout: Optional[Tuple[int, int]],
    figsize: Optional[Tuple[float, float]],
    dpi: int,
    label_position: str,
    label_offset: Tuple[float, float],
    label_fontsize: int,
    label_fontweight: str,
    label_color: str,
    label_style: str,
    spacing: float,
    background_color: str,
    a4_width_constraint: bool,
    max_width_inches: float,
) -> None:
    """Handle simple uniform grid layout."""

    # Convert paths to Path objects
    image_paths = [Path(p) for p in image_paths]

    # Check that all files exist
    for path in image_paths:
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

    n_images = len(image_paths)

    # Determine optimal layout if not provided
    if layout is None:
        layout = _determine_optimal_layout(n_images)

    rows, cols = layout

    # Load images to determine aspect ratios if figsize not provided
    if figsize is None:
        images = [mpimg.imread(str(p)) for p in image_paths]
        figsize = _calculate_optimal_figsize(
            images, rows, cols, a4_width_constraint, max_width_inches
        )
    elif a4_width_constraint and figsize[0] > max_width_inches:
        # Scale down to fit A4 width
        scale = max_width_inches / figsize[0]
        figsize = (max_width_inches, figsize[1] * scale)
        print(f"⚠ Figure width scaled to fit A4: {figsize[0]:.2f}″ × {figsize[1]:.2f}″")

    # Create figure
    fig = plt.figure(figsize=figsize, facecolor=background_color)

    # Create grid spec with spacing
    gs = GridSpec(
        rows,
        cols,
        figure=fig,
        hspace=spacing,
        wspace=spacing,
        left=0.02,
        right=0.98,
        top=0.98,
        bottom=0.02,
    )

    # Generate labels
    labels = _generate_labels(n_images, label_style)

    # Add each image to the grid
    for idx, (img_path, label) in enumerate(zip(image_paths, labels)):
        row = idx // cols
        col = idx % cols

        ax = fig.add_subplot(gs[row, col])

        # Load and display image
        img = mpimg.imread(str(img_path))
        ax.imshow(img)
        ax.axis("off")

        # Add label
        x, y = _get_label_position(label_position, label_offset)
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=label_fontsize,
            fontweight=label_fontweight,
            color=label_color,
            verticalalignment="top" if "top" in label_position else "bottom",
            horizontalalignment="left" if "left" in label_position else "right",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.7
            ),
        )

    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=background_color)
    plt.close()

    print(f"✓ Assembled figure saved to: {output_path}")
    print(f"  - Layout: {rows}×{cols}")
    print(f"  - Resolution: {dpi} DPI")
    print(f"  - Size: {figsize[0]:.1f}″ × {figsize[1]:.1f}″")


def _assemble_advanced_layout(
    panels: List[Dict[str, Any]],
    output_path: str,
    figsize: Optional[Tuple[float, float]],
    dpi: int,
    label_position: str,
    label_offset: Tuple[float, float],
    label_fontsize: int,
    label_fontweight: str,
    label_color: str,
    label_style: str,
    spacing: float,
    background_color: str,
    a4_width_constraint: bool,
    max_width_inches: float,
) -> None:
    """Handle advanced non-uniform grid layout with spanning."""

    # Validate panels
    for i, panel in enumerate(panels):
        if "image" not in panel or "row" not in panel or "col" not in panel:
            raise ValueError(f"Panel {i} must contain 'image', 'row', and 'col' keys")

        path = Path(panel["image"])
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

    # Determine grid size
    max_row = max(p["row"] + p.get("rowspan", 1) for p in panels)
    max_col = max(p["col"] + p.get("colspan", 1) for p in panels)

    # Load images for figsize calculation if needed
    if figsize is None:
        images = [mpimg.imread(str(panel["image"])) for panel in panels]
        figsize = _calculate_optimal_figsize(
            images, max_row, max_col, a4_width_constraint, max_width_inches
        )
    elif a4_width_constraint and figsize[0] > max_width_inches:
        scale = max_width_inches / figsize[0]
        figsize = (max_width_inches, figsize[1] * scale)
        print(f"⚠ Figure width scaled to fit A4: {figsize[0]:.2f}″ × {figsize[1]:.2f}″")

    # Create figure
    fig = plt.figure(figsize=figsize, facecolor=background_color)

    # Create grid spec with spacing
    gs = GridSpec(
        max_row,
        max_col,
        figure=fig,
        hspace=spacing,
        wspace=spacing,
        left=0.02,
        right=0.98,
        top=0.98,
        bottom=0.02,
    )

    # Generate automatic labels
    auto_labels = _generate_labels(len(panels), label_style)

    # Add each panel
    for idx, panel in enumerate(panels):
        row = panel["row"]
        col = panel["col"]
        rowspan = panel.get("rowspan", 1)
        colspan = panel.get("colspan", 1)

        # Create subplot with spanning
        ax = fig.add_subplot(gs[row : row + rowspan, col : col + colspan])

        # Load and display image
        img = mpimg.imread(str(panel["image"]))
        ax.imshow(img)
        ax.axis("off")

        # Determine label (custom or automatic)
        label = panel.get("label", auto_labels[idx])

        # Add label
        x, y = _get_label_position(label_position, label_offset)
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=label_fontsize,
            fontweight=label_fontweight,
            color=label_color,
            verticalalignment="top" if "top" in label_position else "bottom",
            horizontalalignment="left" if "left" in label_position else "right",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.7
            ),
        )

    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=background_color)
    plt.close()

    print(f"✓ Assembled figure saved to: {output_path}")
    print(f"  - Layout: {max_row}×{max_col} grid with spanning")
    print(f"  - Resolution: {dpi} DPI")
    print(f"  - Size: {figsize[0]:.1f}″ × {figsize[1]:.1f}″")


def _determine_optimal_layout(n_images: int) -> Tuple[int, int]:
    """
    Determine optimal grid layout for n images.
    Tries to keep aspect ratio close to 4:3 or 16:9.
    """
    if n_images == 1:
        return (1, 1)
    elif n_images == 2:
        return (1, 2)
    elif n_images == 3:
        return (1, 3)
    elif n_images == 4:
        return (2, 2)
    elif n_images <= 6:
        return (2, 3)
    elif n_images <= 9:
        return (3, 3)
    elif n_images <= 12:
        return (3, 4)
    else:
        # For larger numbers, try to keep roughly square
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
        return (rows, cols)


def _calculate_optimal_figsize(
    images: List[np.ndarray],
    rows: int,
    cols: int,
    a4_constraint: bool = True,
    max_width: float = 8.27,
) -> Tuple[float, float]:
    """
    Calculate optimal figure size based on image dimensions.
    Respects A4 width constraint if enabled.
    """
    # Get average aspect ratio
    aspect_ratios = [img.shape[1] / img.shape[0] for img in images]
    avg_aspect = np.mean(aspect_ratios)

    # Base size per subplot (in inches)
    base_width = 4.0
    base_height = base_width / avg_aspect

    # Total figure size
    width = base_width * cols
    height = base_height * rows

    # Apply A4 width constraint first if enabled
    if a4_constraint and width > max_width:
        scale = max_width / width
        width = max_width
        height *= scale

    # Limit maximum size (after A4 constraint)
    max_size = 20
    if width > max_size or height > max_size:
        scale = max_size / max(width, height)
        width *= scale
        height *= scale

    return (width, height)


def _generate_labels(n: int, style: str) -> List[str]:
    """Generate panel labels based on style."""
    if style == "A":
        return [chr(65 + i) for i in range(n)]  # A, B, C, ...
    elif style == "a":
        return [chr(97 + i) for i in range(n)]  # a, b, c, ...
    elif style == "(A)":
        return [f"({chr(65 + i)})" for i in range(n)]  # (A), (B), (C), ...
    elif style == "A)":
        return [f"{chr(65 + i)})" for i in range(n)]  # A), B), C), ...
    else:
        return [chr(65 + i) for i in range(n)]


def _get_label_position(
    position: str, offset: Tuple[float, float]
) -> Tuple[float, float]:
    """Get label coordinates based on position string."""
    x, y = offset

    if position == "top-left":
        return (x, 1 - (1 - y))
    elif position == "top-right":
        return (1 - x, 1 - (1 - y))
    elif position == "bottom-left":
        return (x, y - 1)
    elif position == "bottom-right":
        return (1 - x, y - 1)
    else:
        return (x, 1 - (1 - y))  # default to top-left
