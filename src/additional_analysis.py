import pandas as pd


def medication_usage(measurements):
    meds = [
        "metformin",
        "glimepiride",
        "glipizide",
        "glyburide",
        "pioglitazone",
        "rosiglitazone",
        "insulin"
    ]

    results = {}

    for m in meds:
        active = measurements[m].isin(["Steady", "Up", "Down"]).sum()
        results[m] = active

    df = pd.DataFrame.from_dict(results, orient="index", columns=["count"])
    df = df.sort_values("count", ascending=False)

    return df


def age_vs_medications(measurements):
    df = measurements.copy()

    df["age_numeric"] = df["age"].str.extract(r"(\d+)").astype(float)

    result = df.groupby("age_numeric")["num_medications"].mean().reset_index()
    result = result.sort_values("age_numeric")

    return result

def age_vs_insulin(measurements):

    df = measurements.copy()

    df["age_numeric"] = df["age"].str.extract(r"(\d+)").astype(float)

    df["insulin_used"] = df["insulin"].isin(["Steady", "Up", "Down"])

    result = (
        df.groupby("age_numeric")["insulin_used"]
        .mean()
        .reset_index()
    )

    result["insulin_percent"] = result["insulin_used"] * 100

    return result