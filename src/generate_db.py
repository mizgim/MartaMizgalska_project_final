import sqlite3
import numpy as np
from datetime import datetime, timedelta
from . import config


def generate_database(db_path):
    rng = np.random.default_rng(config.RANDOM_SEED)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
    DROP TABLE IF EXISTS patients;
    DROP TABLE IF EXISTS visits;
    DROP TABLE IF EXISTS measurements;

    CREATE TABLE patients (
        patient_id INTEGER PRIMARY KEY,
        sex TEXT,
        birth_year INTEGER
    );

    CREATE TABLE visits (
        visit_id INTEGER PRIMARY KEY,
        patient_id INTEGER,
        visit_date TEXT
    );

    CREATE TABLE measurements (
        measurement_id INTEGER PRIMARY KEY,
        visit_id INTEGER,
        patient_id INTEGER,
        kind TEXT,
        value REAL,
        measured_at TEXT
    );
    """)

    visit_id = 1
    measurement_id = 1

    for pid in range(1, config.N_PATIENTS + 1):
        sex = rng.choice(["M", "K"])
        birth_year = rng.integers(1940, 2005)

        cur.execute(
            "INSERT INTO patients VALUES (?, ?, ?)",
            (pid, sex, birth_year),
        )

        n_visits = rng.integers(config.VISITS_MIN, config.VISITS_MAX)

        for i in range(n_visits):
            visit_date = datetime.now() - timedelta(days=int(rng.integers(0, 365)))

            cur.execute(
                "INSERT INTO visits VALUES (?, ?, ?)",
                (visit_id, pid, visit_date.date().isoformat()),
            )

            values = {
                "BP_SYS": rng.normal(125, 15),
                "BP_DIA": rng.normal(80, 10),
                "HR": rng.normal(75, 12),
                "TEMP": rng.normal(36.8, 0.3),
                "BMI": rng.normal(27, 4),
            }

            for kind in config.KINDS:
                if rng.random() < config.MISSING_RATE:
                    continue

                cur.execute(
                    "INSERT INTO measurements VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        measurement_id,
                        visit_id,
                        pid,
                        kind,
                        float(values[kind]),
                        visit_date.isoformat(),
                    ),
                )
                measurement_id += 1

            visit_id += 1

    conn.commit()
    conn.close()