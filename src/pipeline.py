import numpy as np
import pandas as pd
from pathlib import Path

from . import config
from .generate_db import generate_database
from .load_data import load_measurements
from .build_features import build_feature_matrix
from .normalize import normalize_matrix
from .similarity import compute_distance_matrix, top_k_neighbors
from .pca_analysis import compute_pca_2d
from .stability import stability_jaccard
from .stats import compute_input_stats   # <-- DODAJ IMPORT


def run_pipeline():
    base_path = Path(__file__).resolve().parents[1]

    db_path = base_path / "data" / "processed" / "patients.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    results_tables = base_path / "results" / "tables"   # <-- DODAJ
    results_tables.mkdir(parents=True, exist_ok=True)   # <-- DODAJ

    # 1) Generowanie bazy
    generate_database(db_path)

    # 2) Wczytanie danych
    measurements = load_measurements(db_path)

    # 3) Statystyki wejścia -> CSV
    input_stats = compute_input_stats(measurements)
    input_stats.to_csv(results_tables / "input_stats.csv", index=False)

    # 4) Budowa macierzy cech
    X = build_feature_matrix(measurements, aggregation="mean")

    # 5) Normalizacja
    X_norm = normalize_matrix(X, method="zscore")

    # 6) Macierz odległości
    D = compute_distance_matrix(X_norm, metric="euclidean")

    # 7) Najbliżsi sąsiedzi
    neighbors = top_k_neighbors(D, k=config.TOP_K_NEIGHBORS)

    # 8) PCA (na razie tylko liczymy; zapis CSV dodamy później, jeśli chcesz)
    coords = compute_pca_2d(X_norm)

    # 9) Stabilność: mean vs median
    X_median = build_feature_matrix(measurements, aggregation="median")
    X_median = normalize_matrix(X_median, method="zscore")
    D_median = compute_distance_matrix(X_median, metric="euclidean")
    neighbors_median = top_k_neighbors(D_median, k=config.TOP_K_NEIGHBORS)

    rng = np.random.default_rng(config.RANDOM_SEED)
    sample_ids = rng.choice(list(neighbors.keys()), size=20, replace=False)

    stability = stability_jaccard(neighbors, neighbors_median, sample_ids)

    # 10) Zapis stabilności -> CSV
    stability_df = pd.DataFrame([{
        "porownanie": "agregacja: mean vs median (zscore + euclidean)",
        "top_k": config.TOP_K_NEIGHBORS,
        "n_query": len(sample_ids),
        "jaccard_mean": stability
    }])
    stability_df.to_csv(results_tables / "stability_jaccard.csv", index=False)

    print("Średnia stabilność (Jaccard):", stability)
    print("Pipeline zakończony poprawnie.")