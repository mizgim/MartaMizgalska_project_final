import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform


def compute_distance_matrix(X, metric="euclidean"):
    distances = squareform(pdist(X.values, metric=metric))
    return pd.DataFrame(distances, index=X.index, columns=X.index)


def top_k_neighbors(D, k=10):
    neighbors = {}
    for pid in D.index:
        row = D.loc[pid].copy()
        row.loc[pid] = np.inf
        neighbors[pid] = row.nsmallest(k).index.tolist()
    return neighbors