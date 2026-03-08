import pandas as pd
import numpy as np


def encode_age(age_value):
    if pd.isna(age_value):
        return np.nan

    age_value = str(age_value).strip()

    if age_value.startswith("[") and age_value.endswith(")"):
        age_value = age_value[1:-1]
        start, end = age_value.split("-")
        return (int(start) + int(end)) / 2

    return np.nan


def build_feature_matrix(df, aggregation="mean"):
    data = df.copy()

    # Kodowanie age
    data["age_numeric"] = data["age"].apply(encode_age)

    # Kodowanie gender
    gender_map = {
        "Male": 1,
        "Female": 0
    }
    data["gender_numeric"] = data["gender"].map(gender_map)

    # Kodowanie max_glu_serum
    glu_map = {
        "None": 0,
        "Norm": 1,
        ">200": 2,
        ">300": 3
    }
    data["max_glu_serum_numeric"] = data["max_glu_serum"].map(glu_map)

    # Kodowanie A1Cresult
    a1c_map = {
        "None": 0,
        "Norm": 1,
        ">7": 2,
        ">8": 3
    }
    data["A1Cresult_numeric"] = data["A1Cresult"].map(a1c_map)

    # Kodowanie leków
    med_map = {
        "No": 0,
        "Steady": 1,
        "Up": 2,
        "Down": 3
    }

    medication_columns = [
        "metformin",
        "glimepiride",
        "glipizide",
        "glyburide",
        "pioglitazone",
        "rosiglitazone",
        "insulin"
    ]

    for col in medication_columns:
        data[col] = data[col].map(med_map)

    # encoding dla race
    race_dummies = pd.get_dummies(data["race"], prefix="race")

    # Kolumny liczbowe
    numeric_cols = [
        "age_numeric",
        "gender_numeric",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
        "max_glu_serum_numeric",
        "A1Cresult_numeric"
    ] + medication_columns

    features = pd.concat(
        [
            data[["encounter_id"] + numeric_cols],
            race_dummies
        ],
        axis=1
    )

    features = features.set_index("encounter_id")

    # Zamiana na liczby i uzupełnienie braków
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.fillna(0)

    return features