from sklearn.decomposition import PCA
import pandas as pd


def compute_pca_2d(X):

    pca = PCA(n_components=2)

    coords = pca.fit_transform(X)

    coords_df = pd.DataFrame(
        coords,
        index=X.index,
        columns=["PC1", "PC2"]
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=["PC1", "PC2"]
    )

    return coords_df, loadings