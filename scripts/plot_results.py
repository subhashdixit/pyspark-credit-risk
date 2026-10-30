"""Visualize real held-out metrics and Spark-to-serving architecture."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position

ROOT = Path(__file__).resolve().parents[1]


def plot() -> None:
    """Render confusion counts with explicit historical-source labels."""
    report = json.loads((ROOT / "models/metrics.json").read_text())
    matrix = np.array(report["test"]["confusion_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].imshow(matrix, cmap="Greens")
    for row in range(2):
        for column in range(2):
            axes[0].text(
                column, row, str(matrix[row, column]), ha="center", va="center"
            )
    axes[0].set(
        xticks=[0, 1],
        yticks=[0, 1],
        xlabel="Predicted source label",
        ylabel="Actual source label",
        title="Held-out UCI default labels",
    )
    names = list(report["validation_candidates"])
    axes[1].barh(
        names,
        [report["validation_candidates"][name]["average_precision"] for name in names],
        color="#24664e",
    )
    axes[1].set(
        xlim=(0, 1),
        xlabel="Validation average precision",
        title="Model selection used validation only",
    )
    fig.savefig(ROOT / "docs/images/evaluation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot()
