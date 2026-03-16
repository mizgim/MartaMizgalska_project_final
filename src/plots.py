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

def plot_pca_loadings(loadings, out_path, top_n=10):

    import numpy as np

    importance = np.sqrt(loadings["PC1"]**2 + loadings["PC2"]**2)

    top_features = importance.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8,6))

    ax.barh(
        top_features.index[::-1],
        top_features.values[::-1]
    )

    ax.set_title("Najważniejsze zmienne w PCA")
    ax.set_xlabel("Wpływ na strukturę danych")

    ax.grid(True, linestyle="--", alpha=0.4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)

    plt.close(fig)

def plot_medication_usage(df, out_path):

    fig, ax = plt.subplots(figsize=(8,6))

    ax.barh(df.index[::-1], df["count"][::-1])

    ax.set_title("Najczęściej stosowane leki w hospitalizacjach")
    ax.set_xlabel("Liczba hospitalizacji")

    ax.grid(True, linestyle="--", alpha=0.4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)

    plt.close(fig)

def plot_medication_usage(df, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.barh(df.index[::-1], df["count"][::-1])

    ax.set_title("Najczęściej stosowane leki")
    ax.set_xlabel("Liczba hospitalizacji")
    ax.set_ylabel("Lek")

    ax.grid(True, linestyle="--", alpha=0.4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_age_vs_medications(df, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(df["age_numeric"], df["num_medications"], marker="o")

    ax.set_title("Średnia liczba leków a wiek pacjenta")
    ax.set_xlabel("Wiek")
    ax.set_ylabel("Średnia liczba leków")

    ax.grid(True, linestyle="--", alpha=0.4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def plot_age_vs_insulin(df, out_path):

    fig, ax = plt.subplots(figsize=(8,6))

    ax.plot(
        df["age_numeric"],
        df["insulin_percent"],
        marker="o"
    )

    ax.set_title("Odsetek insulinoterapii vs wiek pacjenta")
    ax.set_xlabel("Wiek")
    ax.set_ylabel("Pacjenci leczeni insuliną (%)")

    ax.grid(True, linestyle="--", alpha=0.4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)

    plt.close(fig)