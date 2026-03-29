from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as SklearnPCA

from . import config
from .generate_db import generate_database
from .stats import compute_input_stats
from .load_data import load_measurements
from .build_features import build_feature_matrix
from .normalize import normalize_matrix
from .similarity import compute_knn, knn_to_dataframe
from .pca_analysis import compute_pca_2d
from .stability import stability_jaccard
from .plots import plot_pca, plot_pca_insulin, plot_pca_loadings
from .plots import plot_medication_usage, plot_age_vs_medications, plot_age_vs_insulin
from .additional_analysis import medication_usage, age_vs_medications, age_vs_insulin


def run_pipeline(sample_size=None, status_callback=None):

    def update(msg, pct):
        print(msg)
        if status_callback:
            status_callback(msg, pct)

    base_path = Path(__file__).resolve().parents[1]
    csv_path = base_path / "data" / "diabetic_data.csv"
    db_path = base_path / "data" / "processed" / "patients.db"
    suffix = "sample" if sample_size else "full"

    results_tables = Path(f"results/{suffix}/tables")
    results_plots = Path(f"results/{suffix}/plots")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    results_tables.mkdir(parents=True, exist_ok=True)
    results_plots.mkdir(parents=True, exist_ok=True)

    # 1. Import CSV -> SQLite
    update("Sprawdzam bazę danych...", 5)
    if not db_path.exists():
        update("Tworzę bazę danych...", 8)
        generate_database(csv_path, db_path)

    # 2. Wczytanie danych
    update("Wczytuję dane z bazy...", 15)
    measurements = load_measurements(db_path)
    if sample_size:
        measurements = measurements.sample(n=sample_size, random_state=42)
    input_stats = compute_input_stats(measurements)
    input_stats.to_csv(results_tables / "input_stats.csv", index=False)

    # 3. Dodatkowe analizy
    update("Analizuję farmakoterapię...", 25)
    med_usage = medication_usage(measurements)
    med_usage.to_csv(results_tables / "medication_usage.csv")
    plot_medication_usage(med_usage, results_plots / "medication_usage.png")

    age_meds = age_vs_medications(measurements)
    age_meds.to_csv(results_tables / "age_vs_medications.csv")
    plot_age_vs_medications(age_meds, results_plots / "age_vs_medications.png")

    age_insulin = age_vs_insulin(measurements)
    plot_age_vs_insulin(age_insulin, results_plots / "age_vs_insulin.png")

    # 4. Budowa macierzy cech
    update("Buduję macierz cech...", 35)
    X = build_feature_matrix(measurements)
    X.to_csv(results_tables / "feature_matrix.csv")

    # 5. Normalizacja
    update("Normalizuję dane...", 45)
    X_norm = normalize_matrix(X, method="zscore")

    # 6. Redukcja wymiarów przed kNN (rekomendacja: PCA + kNN)
    update("Redukuję wymiary przed kNN (PCA 10 komponentów)...", 50)
    n_components = min(10, X_norm.shape[1])
    pca_pre = SklearnPCA(n_components=n_components, random_state=config.RANDOM_SEED)
    X_reduced = pca_pre.fit_transform(X_norm)
    explained = pca_pre.explained_variance_ratio_.sum()
    print(f"PCA przed kNN: {n_components} komponentów, wyjaśniona wariancja: {explained:.2%}")

    # 7. kNN na zredukowanych danych
    update("Obliczam k-nearest neighbors...", 60)
    distances, indices = compute_knn(
        X_reduced,
        n_neighbors=config.TOP_K_NEIGHBORS,
        algorithm="ball_tree",
        metric="euclidean"
    )
    neighbors_df = knn_to_dataframe(indices, X.index)
    neighbors_df.to_csv(results_tables / "neighbors.csv", index=False)

    # 8. PCA 2D do wizualizacji
    update("Obliczam PCA 2D do wizualizacji...", 70)
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
    plot_pca_loadings(loadings, results_plots / "pca_loadings.png")
    plot_pca(coords_out, results_plots / "pca_plot.png",
             title="PCA hospitalizacji diabetologicznych")
    plot_pca_insulin(coords_out, results_plots / "pca_insulin.png",
                     title="PCA hospitalizacji (kolor: insulin)")

    # 9. Stabilność (Jaccard) - tylko dla próbki
    update("Obliczam stabilność (Jaccard)...", 85)
    if sample_size:
        X_minmax = normalize_matrix(X, method="minmax")
        X_reduced_minmax = pca_pre.transform(X_minmax)  # ten sam PCA, inna normalizacja

        _, indices_minmax = compute_knn(
            X_reduced_minmax,
            n_neighbors=config.TOP_K_NEIGHBORS,
            algorithm="ball_tree",
            metric="euclidean"
        )

        idx = np.array(X.index)
        neighbors_zscore = {idx[i]: idx[indices[i]].tolist() for i in range(len(indices))}
        neighbors_minmax = {idx[i]: idx[indices_minmax[i]].tolist() for i in range(len(indices_minmax))}

        rng = np.random.default_rng(config.RANDOM_SEED)
        query_n = min(100, len(neighbors_zscore))
        query_ids = rng.choice(list(neighbors_zscore.keys()), size=query_n, replace=False)
        stability = stability_jaccard(neighbors_zscore, neighbors_minmax, query_ids)

        stability_df = pd.DataFrame([{
            "porownanie": "normalizacja: zscore vs minmax (PCA 10D + euclidean)",
            "top_k": config.TOP_K_NEIGHBORS,
            "n_query": len(query_ids),
            "jaccard_mean": stability
        }])
    else:
        stability_df = pd.DataFrame([{
            "porownanie": "normalizacja: zscore vs minmax (PCA 10D + euclidean)",
            "top_k": config.TOP_K_NEIGHBORS,
            "n_query": 0,
            "jaccard_mean": None
        }])

    stability_df.to_csv(results_tables / "stability_jaccard.csv", index=False)

    update("Zapisuję wyniki...", 95)
    print("Liczba hospitalizacji:", len(X))
    print("Liczba cech:", X.shape[1])
    print(f"Przestrzeń kNN: PCA {n_components}D (z {X.shape[1]} cech)")

    if sample_size:
        print("Średnia stabilność (Jaccard):", stability)
    else:
        print("Stabilność nie była liczona dla pełnego zbioru danych.")

    print("Pipeline zakończony poprawnie.")