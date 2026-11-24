"""Plot generation for reports.

Per spec §12.5: matplotlib figures (PDF+PNG), consistent style, colorblind-safe.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adg.core.logging import get_logger

log = get_logger(__name__)

# Colorblind-safe palette (Wong 2011)
COLORS = {
    "blue": "#0173B2",
    "orange": "#DE8F05",
    "green": "#029E73",
    "red": "#CC78BC",
    "purple": "#CA9161",
    "brown": "#949494",
    "gray": "#ECE133",
}

# Default plot style
PLOT_STYLE = {
    "figure.figsize": (8, 6),
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 100,
}


def setup_plot_style() -> None:
    """Set up consistent matplotlib style."""
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(PLOT_STYLE)


def save_figure(
    fig: plt.Figure,
    output_path: Path,
    formats: list[str] | None = None,
) -> None:
    """Save figure in multiple formats.

    Args:
        fig: Matplotlib figure
        output_path: Output path (without extension)
        formats: List of formats to save (pdf, png)
    """
    if formats is None:
        formats = ["pdf", "png"]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        path = output_path.with_suffix(f".{fmt}")
        fig.savefig(path, bbox_inches="tight", dpi=300 if fmt == "png" else None)
        log.info("figure_saved", path=str(path), format=fmt)


def plot_confusion_matrix(
    cm: list[list[int]],
    labels: list[str],
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot confusion matrix.

    Per spec §12.5: Confusion matrix for 3-way classification.

    Args:
        cm: Confusion matrix (3x3)
        labels: Class labels (ANSWER, ASK, ABSTAIN)
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(6, 5))

    # Normalize to show percentages
    cm_array = np.array(cm)
    cm_norm = cm_array.astype("float") / cm_array.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)  # Handle division by zero

    # Plot heatmap
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Proportion", rotation=270, labelpad=15)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Add text annotations
    # Threshold for text color contrast
    text_color_threshold = 0.5

    for i in range(len(labels)):
        for j in range(len(labels)):
            count = cm_array[i, j]
            percent = cm_norm[i, j]
            ax.text(
                j,
                i,
                f"{count}\n({percent:.1%})",
                ha="center",
                va="center",
                color="white" if percent > text_color_threshold else "black",
                fontsize=9,
            )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_risk_coverage_curve(
    coverages: list[float],
    accuracies: list[float],
    method_names: list[str],
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot risk-coverage curves.

    Per spec §12.5: Accuracy vs coverage for selective answering.

    Args:
        coverages: List of coverage values per method
        accuracies: List of accuracy values per method
        method_names: Method names
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = list(COLORS.values())

    for i, (cov, acc, name) in enumerate(zip(coverages, accuracies, method_names, strict=False)):
        color = colors[i % len(colors)]
        ax.plot(cov, acc, marker="o", label=name, color=color, linewidth=2)

    ax.set_xlabel("Coverage (fraction of queries answered)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Risk-Coverage Curves")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_budget_sweep(
    budgets: list[int],
    metrics: dict[str, list[float]],
    metric_name: str = "F1",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot metric vs budget (number of samples).

    Per spec §12.5: Budget sweep showing metric vs samples.

    Args:
        budgets: List of budget values (number of samples)
        metrics: Dict of method_name -> metric values
        metric_name: Name of metric being plotted
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = list(COLORS.values())

    for i, (method, values) in enumerate(metrics.items()):
        color = colors[i % len(colors)]
        ax.plot(budgets, values, marker="o", label=method, color=color, linewidth=2)

    ax.set_xlabel("Number of Samples")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} vs Budget")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_feature_scatter(
    p1_values: list[float],
    delta_values: list[float],
    gold_actions: list[str],
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot feature scatter (p1 vs Δ).

    Per spec §12.5: Feature scatter colored by gold action,
    visually shows ambiguity vs ignorance separation.

    Args:
        p1_values: p1 (largest cluster share) values
        delta_values: Δ (separation) values
        gold_actions: Gold action labels for coloring
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color by action
    action_colors = {
        "ANSWER": COLORS["blue"],
        "ASK": COLORS["orange"],
        "ABSTAIN": COLORS["red"],
    }

    for action in ["ANSWER", "ASK", "ABSTAIN"]:
        mask = [a == action for a in gold_actions]
        x = [p1 for p1, m in zip(p1_values, mask, strict=False) if m]
        y = [d for d, m in zip(delta_values, mask, strict=False) if m]

        if x:  # Only plot if there are points
            ax.scatter(
                x,
                y,
                label=action,
                color=action_colors[action],
                alpha=0.6,
                s=50,
            )

    ax.set_xlabel("p1 (Largest Cluster Share)")
    ax.set_ylabel("Δ (Separation)")
    ax.set_title("Feature Space: Ambiguity vs Ignorance")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_cluster_histogram(
    cluster_counts: list[int],
    gold_actions: list[str],
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot cluster count histogram by gold action.

    Per spec §12.5: Cluster-count histograms by gold action.

    Args:
        cluster_counts: Number of clusters per example
        gold_actions: Gold action labels
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Separate by action
    data_by_action = {
        "ANSWER": [],
        "ASK": [],
        "ABSTAIN": [],
    }

    for count, action in zip(cluster_counts, gold_actions, strict=False):
        if action in data_by_action:
            data_by_action[action].append(count)

    # Plot histograms
    action_colors = {
        "ANSWER": COLORS["blue"],
        "ASK": COLORS["orange"],
        "ABSTAIN": COLORS["red"],
    }

    bins = range(1, max(cluster_counts) + 2)

    for action in ["ANSWER", "ASK", "ABSTAIN"]:
        if data_by_action[action]:
            ax.hist(
                data_by_action[action],
                bins=bins,
                label=action,
                color=action_colors[action],
                alpha=0.6,
                edgecolor="black",
            )

    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Count")
    ax.set_title("Cluster Count Distribution by Gold Action")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


__all__ = [
    "plot_confusion_matrix",
    "plot_risk_coverage_curve",
    "plot_budget_sweep",
    "plot_feature_scatter",
    "plot_cluster_histogram",
    "setup_plot_style",
    "save_figure",
]
