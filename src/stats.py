import pandas as pd


def compute_input_stats(df):

    stats = []

    for col in df.columns:

        series = df[col]

        stat = {
            "column": col,
            "dtype": str(series.dtype),
            "n_unique": series.nunique(),
            "missing": series.isna().sum()
        }

        if pd.api.types.is_numeric_dtype(series):

            stat.update({
                "mean": series.mean(),
                "median": series.median(),
                "min": series.min(),
                "max": series.max(),
                "std": series.std()
            })

        stats.append(stat)

    return pd.DataFrame(stats)