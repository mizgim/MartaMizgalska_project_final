from pathlib import Path
import numpy as np
import pandas as pd

from . import config
from .generate_db import generate_database
from .load_data import load_measurements
from .build_features import build_feature_matrix
from .normalize import normalize_matrix
from .similarity import compute_distance_matrix, top_k_neighbors
from .pca_analysis import compute_pca_2d
from .stability import stability_jaccard
from .plots import plot_pca
from .plots import plot_pca, plot_pca_insulin
from .plots import plot_pca, plot_pca_insulin, plot_pca_loadings

def run_pipeline():
    base_path = Path(__file__).resolve().parents[1]

    csv_path = base_path / "data" / "diabetic_data.csv"
    db_path = base_path / "data" / "processed" / "patients.db"
    results_tables = base_path / "results" / "tables"
    results_plots = base_path / "results" / "plots"

    db_path.parent.mkdir(parents=True, exist_ok=True)
    results_tables.mkdir(parents=True, exist_ok=True)

    # 1. Import CSV -> SQLite
    generate_database(csv_path, db_path)

    # 2. Wczytanie danych po joinach
    measurements = load_measurements(db_path)

    # Próbka robocza ze względu na rozmiar danych
    sample_size = min(5000, len(measurements))
    measurements = measurements.sample(n=sample_size, random_state=config.RANDOM_SEED)

    # 3. Budowa macierzy cech
    X = build_feature_matrix(measurements)

    # zapis macierzy cech
    X.to_csv(results_tables / "feature_matrix.csv")

    # 4. Normalizacja
    X_norm = normalize_matrix(X, method="zscore")

    # 5. Macierz odległości
    D = compute_distance_matrix(X_norm, metric="euclidean")

    # 6. Sąsiedzi
    neighbors = top_k_neighbors(D, k=config.TOP_K_NEIGHBORS)

    # 7. PCA
    coords, loadings = compute_pca_2d(X_norm)

    coords_out = coords.reset_index().rename(columns={"index": "encounter_id"})
    coords_out["readmitted"] = measurements.set_index("encounter_id").loc[
        coords_out["encounter_id"], "readmitted"
    ].values

    coords_out["insulin"] = measurements.set_index("encounter_id").loc[
        coords_out["encounter_id"], "insulin"
    ].values

    coords_out.to_csv(results_tables / "pca_coords.csv", index=False)
    loadings.to_csv(results_tables / "pca_loadings.csv")
    plot_pca_loadings(
        loadings,
        results_plots / "pca_loadings.png"
    )

    results_plots = base_path / "results" / "plots"
    results_plots.mkdir(parents=True, exist_ok=True)

    plot_pca(
        coords_out,
        results_plots / "pca_plot.png",
        title="PCA hospitalizacji diabetologicznych"
    )

    plot_pca_insulin(
        coords_out,
        results_plots / "pca_insulin.png",
        title="PCA hospitalizacji (kolor: insulin)"
    )

    # 8. Stabilność: zscore vs minmax
    X_minmax = normalize_matrix(X, method="minmax")
    D_minmax = compute_distance_matrix(X_minmax, metric="euclidean")
    neighbors_minmax = top_k_neighbors(D_minmax, k=config.TOP_K_NEIGHBORS)

    rng = np.random.default_rng(config.RANDOM_SEED)
    sample_size = min(100, len(neighbors))
    sample_ids = rng.choice(list(neighbors.keys()), size=sample_size, replace=False)

    stability = stability_jaccard(neighbors, neighbors_minmax, sample_ids)

    stability_df = pd.DataFrame([{
        "porownanie": "normalizacja: zscore vs minmax (euclidean)",
        "top_k": config.TOP_K_NEIGHBORS,
        "n_query": len(sample_ids),
        "jaccard_mean": stability
    }])
    stability_df.to_csv(results_tables / "stability_jaccard.csv", index=False)

    print("Liczba hospitalizacji:", len(X))
    print("Liczba cech:", X.shape[1])
    print("Średnia stabilność (Jaccard):", stability)
    print("Pipeline zakończony poprawnie.")