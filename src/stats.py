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
                "mean": round(series.mean(), 4),
                "median": series.median(),
                "min": series.min(),
                "max": series.max(),
                "std": round(series.std(), 4),
                "top_values": None
            })
        else:
            # dla kolumn kategorycznych - top 3 najczęstsze wartości z liczebnością
            top = series.value_counts().head(3)
            top_str = ", ".join([f"{v}: {c}" for v, c in top.items()])
            stat.update({
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
                "std": None,
                "top_values": top_str
            })

        stats.append(stat)

    return pd.DataFrame(stats)