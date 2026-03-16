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