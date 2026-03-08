import sqlite3
import pandas as pd
from pathlib import Path


def generate_database(csv_path, db_path):
    csv_path = Path(csv_path)
    db_path = Path(db_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku CSV: {csv_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Usunięcie tabel, jeśli już istnieją
    cur.executescript("""
    DROP TABLE IF EXISTS patients;
    DROP TABLE IF EXISTS encounters;
    DROP TABLE IF EXISTS diagnoses;
    DROP TABLE IF EXISTS labs;
    DROP TABLE IF EXISTS medications;
    DROP TABLE IF EXISTS outcomes;
    """)

    # Tabela patients
    patients = df[
        ["patient_nbr", "race", "gender", "age", "weight"]
    ].drop_duplicates(subset=["patient_nbr"]).copy()

    patients.to_sql("patients", conn, index=False, if_exists="replace")

    # Tabela encounters
    encounters = df[
        [
            "encounter_id",
            "patient_nbr",
            "admission_type_id",
            "discharge_disposition_id",
            "admission_source_id",
            "time_in_hospital",
            "payer_code",
            "medical_specialty",
            "num_lab_procedures",
            "num_procedures",
            "num_medications",
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
            "number_diagnoses",
        ]
    ].copy()

    encounters.to_sql("encounters", conn, index=False, if_exists="replace")

    # Tabela diagnoses
    diagnoses = df[
        ["encounter_id", "diag_1", "diag_2", "diag_3"]
    ].copy()

    diagnoses.to_sql("diagnoses", conn, index=False, if_exists="replace")

    # Tabela labs
    labs = df[
        ["encounter_id", "max_glu_serum", "A1Cresult"]
    ].copy()

    labs.to_sql("labs", conn, index=False, if_exists="replace")

    # Tabela medications
    medications = df[
        [
            "encounter_id",
            "metformin",
            "repaglinide",
            "nateglinide",
            "chlorpropamide",
            "glimepiride",
            "acetohexamide",
            "glipizide",
            "glyburide",
            "tolbutamide",
            "pioglitazone",
            "rosiglitazone",
            "acarbose",
            "miglitol",
            "troglitazone",
            "tolazamide",
            "examide",
            "citoglipton",
            "insulin",
            "glyburide-metformin",
            "glipizide-metformin",
            "glimepiride-pioglitazone",
            "metformin-rosiglitazone",
            "metformin-pioglitazone",
        ]
    ].copy()

    medications.to_sql("medications", conn, index=False, if_exists="replace")

    # Tabela outcomes
    outcomes = df[
        ["encounter_id", "change", "diabetesMed", "readmitted"]
    ].copy()

    outcomes.to_sql("outcomes", conn, index=False, if_exists="replace")

    conn.commit()
    conn.close()