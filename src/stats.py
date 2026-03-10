import pandas as pd


def compute_input_stats(df):

    stats = []

    for col in df.columns:

        series = df[col]

        stats.append({
            "column": col,
            "dtype": str(series.dtype),
            "n_unique": series.nunique(),
            "missing": series.isna().sum()
        })

    stats_df = pd.DataFrame(stats)

    return stats_df