import pandas as pd
import numpy as np


def normalize_matrix(X, method="minmax"):
    X = X.copy()

    if method == "minmax":
        mins = X.min()
        maxs = X.max()
        denom = (maxs - mins).replace(0, np.nan)
        X = (X - mins) / denom
        X = X.fillna(0)

    elif method == "zscore":
        means = X.mean()
        stds = X.std().replace(0, np.nan)
        X = (X - means) / stds
        X = X.fillna(0)

    else:
        raise ValueError("Nieobsługiwana metoda normalizacji")

    return X