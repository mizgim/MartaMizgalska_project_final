from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_pca(coords: pd.DataFrame, out_path: Path, title: str = "PCA hospitalizacji diabetologicznych"):

    colors = {
        "NO": "steelblue",
        "<30": "red",
        ">30": "orange"
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for label, color in colors.items():
        subset = coords[coords["readmitted"] == label]

        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            label=label,
            alpha=0.6,
            s=18,
            color=color
        )

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    ax.legend(title="Readmitted")

    ax.grid(True, linestyle="--", alpha=0.4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_pca_insulin(coords: pd.DataFrame, out_path: Path, title: str = "PCA hospitalizacji (kolor: insulin)"):

    colors = {
        "No": "steelblue",
        "Steady": "orange",
        "Up": "red",
        "Down": "green"
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for label, color in colors.items():
        subset = coords[coords["insulin"] == label]

        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            label=label,
            alpha=0.6,
            s=18,
            color=color
        )

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    ax.legend(title="Insulin")

    ax.grid(True, linestyle="--", alpha=0.4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)