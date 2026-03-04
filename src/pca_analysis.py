import pandas as pd
from sklearn.decomposition import PCA


def compute_pca_2d(X):
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X.values)

    return pd.DataFrame(
        coords,
        index=X.index,
        columns=["PC1", "PC2"]
    )