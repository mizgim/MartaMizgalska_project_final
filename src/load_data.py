import sqlite3
import pandas as pd


def load_table(db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df


def load_measurements(db_path):
    return load_table(db_path, "measurements")


def load_patients(db_path):
    return load_table(db_path, "patients")


def load_visits(db_path):
    return load_table(db_path, "visits")