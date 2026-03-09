from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_pca(coords: pd.DataFrame, out_path: Path, title: str = "PCA hospitalizacji diabetologicznych"):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        coords["PC1"],
        coords["PC2"],
        alpha=0.6,
        s=18
    )

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle="--", alpha=0.4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)