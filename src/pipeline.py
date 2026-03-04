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


def run_pipeline():
    base_path = Path(__file__).resolve().parents[1]
    db_path = base_path / "data" / "processed" / "patients.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 1️⃣ Generowanie bazy
    generate_database(db_path)

    # 2️⃣ Wczytanie danych
    measurements = load_measurements(db_path)

    # 3️⃣ Budowa macierzy cech
    X = build_feature_matrix(measurements, aggregation="mean")

    # 4️⃣ Normalizacja
    X_norm = normalize_matrix(X, method="zscore")

    # 5️⃣ Macierz odległości
    D = compute_distance_matrix(X_norm, metric="euclidean")

    # 6️⃣ Najbliżsi sąsiedzi
    neighbors = top_k_neighbors(D, k=config.TOP_K_NEIGHBORS)

    # 7️⃣ PCA
    coords = compute_pca_2d(X_norm)

    # 8️⃣ Prosta demonstracja stabilności (porównanie mean vs median)
    X_median = build_feature_matrix(measurements, aggregation="median")
    X_median = normalize_matrix(X_median, method="zscore")
    D_median = compute_distance_matrix(X_median)
    neighbors_median = top_k_neighbors(D_median, k=config.TOP_K_NEIGHBORS)

    rng = np.random.default_rng(config.RANDOM_SEED)
    sample_ids = rng.choice(list(neighbors.keys()), size=20, replace=False)

    stability = stability_jaccard(neighbors, neighbors_median, sample_ids)

    print("Średnia stabilność (Jaccard):", stability)
    print("Pipeline zakończony poprawnie.")