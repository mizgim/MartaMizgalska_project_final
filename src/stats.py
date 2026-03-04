import pandas as pd


def compute_input_stats(measurements_df: pd.DataFrame) -> pd.DataFrame:
    df = measurements_df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    rows = []
    for kind, g in df.groupby("kind"):
        total = len(g)
        missing = int(g["value"].isna().sum())
        gv = g["value"].dropna()

        rows.append({
            "kind": kind,
            "n_records": total,
            "n_missing_value": missing,
            "min": float(gv.min()) if len(gv) else None,
            "max": float(gv.max()) if len(gv) else None,
            "mean": float(gv.mean()) if len(gv) else None,
            "median": float(gv.median()) if len(gv) else None,
        })

    return pd.DataFrame(rows).sort_values("kind")