from sklearn.neighbors import NearestNeighbors
import pandas as pd


def compute_knn(X, n_neighbors=10, algorithm="ball_tree", metric="euclidean"):
    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        metric=metric
    )

    model.fit(X)

    distances, indices = model.kneighbors(X)

    return distances, indices


def knn_to_dataframe(indices, X_index):
    rows = []

    for row_id, neighbors in enumerate(indices):
        encounter_id = X_index[row_id]
        neighbor_ids = [X_index[i] for i in neighbors]

        rows.append({
            "encounter_id": encounter_id,
            "neighbors": neighbor_ids
        })

    return pd.DataFrame(rows)