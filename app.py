import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image

BASE = Path(__file__).resolve().parent
TABLES = BASE / "results" / "tables"
PLOTS = BASE / "results" / "plots"

st.set_page_config(page_title="Analiza hospitalizacji diabetologicznych", layout="wide")

st.title("Analiza hospitalizacji diabetologicznych")
st.caption("Dashboard projektu rocznego – analiza podobieństwa hospitalizacji pacjentów z cukrzycą")

menu = st.sidebar.radio(
    "Wybierz widok",
    [
        "Przegląd danych",
        "PCA",
        "Leki w populacji",
        "Wiek vs liczba leków",
        "Wiek vs insulina",
        "Stabilność",
    ],
)

if menu == "Przegląd danych":
    st.subheader("Podstawowe informacje")
    col1, col2, col3 = st.columns(3)

    feature_matrix = TABLES / "feature_matrix.csv"
    input_stats = TABLES / "input_stats.csv"
    stability = TABLES / "stability_jaccard.csv"

    if feature_matrix.exists():
        df = pd.read_csv(feature_matrix)
        col1.metric("Liczba hospitalizacji", len(df))
        col2.metric("Liczba cech", len(df.columns) - 1)
    else:
        col1.metric("Liczba hospitalizacji", "brak")
        col2.metric("Liczba cech", "brak")

    if stability.exists():
        s = pd.read_csv(stability)
        col3.metric("Średnia stabilność", round(float(s["jaccard_mean"].iloc[0]), 3))
    else:
        col3.metric("Średnia stabilność", "brak")

    if input_stats.exists():
        st.subheader("Statystyki wejścia")
        st.dataframe(pd.read_csv(input_stats), use_container_width=True)

elif menu == "PCA":
    st.subheader("PCA hospitalizacji")
    c1, c2 = st.columns(2)

    p1 = PLOTS / "pca_plot.png"
    p2 = PLOTS / "pca_insulin.png"

    with c1:
        st.markdown("**PCA pokolorowane readmission**")
        if p1.exists():
            st.image(str(p1), use_container_width=True)
        else:
            st.warning("Brak wykresu pca_plot.png")

    with c2:
        st.markdown("**PCA pokolorowane insuliną**")
        if p2.exists():
            st.image(str(p2), use_container_width=True)
        else:
            st.warning("Brak wykresu pca_insulin.png")

elif menu == "Leki w populacji":
    st.subheader("Najczęściej stosowane leki")
    p = PLOTS / "medication_usage.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
    else:
        st.warning("Brak wykresu medication_usage.png")

elif menu == "Wiek vs liczba leków":
    st.subheader("Średnia liczba leków a wiek")
    p = PLOTS / "age_vs_medications.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
    else:
        st.warning("Brak wykresu age_vs_medications.png")

elif menu == "Wiek vs insulina":
    st.subheader("Odsetek insulinoterapii a wiek")
    p = PLOTS / "age_vs_insulin.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
    else:
        st.warning("Brak wykresu age_vs_insulin.png")

elif menu == "Stabilność":
    st.subheader("Stabilność struktury podobieństw")
    pca_loadings = TABLES / "pca_loadings.csv"
    stability = TABLES / "stability_jaccard.csv"
    p = PLOTS / "pca_loadings.png"

    if stability.exists():
        st.markdown("**Wynik Jaccarda**")
        st.dataframe(pd.read_csv(stability), use_container_width=True)

    if p.exists():
        st.markdown("**Najważniejsze zmienne w PCA**")
        st.image(str(p), use_container_width=True)

    if pca_loadings.exists():
        with st.expander("Tabela PCA loadings"):
            st.dataframe(pd.read_csv(pca_loadings), use_container_width=True)