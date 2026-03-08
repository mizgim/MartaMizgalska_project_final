import sqlite3
import pandas as pd


def load_measurements(db_path):
    conn = sqlite3.connect(db_path)

    query = """
    SELECT
        e.encounter_id,
        p.race,
        p.gender,
        p.age,
        e.time_in_hospital,
        e.num_lab_procedures,
        e.num_procedures,
        e.num_medications,
        e.number_outpatient,
        e.number_emergency,
        e.number_inpatient,
        e.number_diagnoses,
        l.max_glu_serum,
        l.A1Cresult,
        m.metformin,
        m.glimepiride,
        m.glipizide,
        m.glyburide,
        m.pioglitazone,
        m.rosiglitazone,
        m.insulin,
        o.readmitted
    FROM encounters e
    JOIN patients p ON e.patient_nbr = p.patient_nbr
    LEFT JOIN labs l ON e.encounter_id = l.encounter_id
    LEFT JOIN medications m ON e.encounter_id = m.encounter_id
    LEFT JOIN outcomes o ON e.encounter_id = o.encounter_id
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df