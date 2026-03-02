import pandas as pd


def build_feature_matrix(measurements_df, aggregation="mean"):
    df = measurements_df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    if aggregation == "mean":
        agg_func = "mean"
    elif aggregation == "median":
        agg_func = "median"
    else:
        raise ValueError("Unsupported aggregation")

    grouped = df.groupby(["patient_id", "kind"])["value"].agg(
        ["min", "max", "std", "count", agg_func]
    )

    grouped = grouped.unstack("kind")
    grouped.columns = [f"{stat}_{kind}" for stat, kind in grouped.columns]

    grouped = grouped.fillna(0)

    return grouped