import streamlit as st
import pandas as pd
from pathlib import Path
from src.pipeline import run_pipeline

st.set_page_config(
    page_title="Analiza hospitalizacji diabetologicznych",
    layout="wide"
)

BASE = Path(__file__).resolve().parent

dataset_type = st.sidebar.radio(
    "Wybierz dane",
    ["full", "sample"],
    key="dataset_radio"
)

menu = st.sidebar.radio(
    "Wybierz sekcję",
    [
        "Przegląd",
        "PCA",
        "Terapia i populacja",
        "Stabilność",
        "Dane wejściowe",
        "Wnioski"
    ],
    key="menu_radio"
)

TABLES = BASE / "results" / dataset_type / "tables"
PLOTS = BASE / "results" / dataset_type / "plots"

st.title("Analiza hospitalizacji diabetologicznych")
st.caption(
    "Interaktywny dashboard projektu rocznego — analiza podobieństwa hospitalizacji pacjentów z cukrzycą"
)

st.info(f"Aktualnie wybrany zbiór: {dataset_type}")

if st.button("Uruchom pipeline"):
    progress = st.progress(0, text="Inicjalizacja...")
    status = st.empty()

    def update_status(msg, pct):
        progress.progress(pct, text=msg)
        status.info(msg)

    if dataset_type == "sample":
        update_status("Wczytuję dane...", 10)
        run_pipeline(sample_size=5000, status_callback=update_status)
    else:
        update_status("Wczytuję dane... (pełny zbiór – może potrwać kilka minut)", 10)
        run_pipeline(sample_size=None, status_callback=update_status)

    progress.progress(100, text="Gotowe!")
    st.success("Pipeline zakończony.")
    st.rerun()

    st.success("Pipeline zakończony. Wyniki zostały odświeżone.")
    st.rerun()


feature_matrix_path = TABLES / "feature_matrix.csv"
input_stats_path = TABLES / "input_stats.csv"
stability_path = TABLES / "stability_jaccard.csv"
loadings_path = TABLES / "pca_loadings.csv"

if feature_matrix_path.exists():
    feature_matrix = pd.read_csv(feature_matrix_path)
else:
    feature_matrix = None

if input_stats_path.exists():
    input_stats = pd.read_csv(input_stats_path)
else:
    input_stats = None

if stability_path.exists():
    stability_df = pd.read_csv(stability_path)
else:
    stability_df = None

if menu == "Przegląd":
    st.subheader("Podstawowe informacje")

    col1, col2, col3 = st.columns(3)

    if feature_matrix is not None:
        col1.metric("Liczba hospitalizacji", len(feature_matrix))
        col2.metric("Liczba cech", len(feature_matrix.columns) - 1)
    else:
        col1.metric("Liczba hospitalizacji", "brak")
        col2.metric("Liczba cech", "brak")

    if stability_df is not None:
        col3.metric(
            "Średnia stabilność (Jaccard)",
            round(float(stability_df["jaccard_mean"].iloc[0]), 3)
        )
    else:
        col3.metric("Średnia stabilność (Jaccard)", "brak")

    st.markdown("---")
    st.markdown(
        """
        ### Cel projektu
        Projekt analizuje podobieństwo hospitalizacji pacjentów z cukrzycą na podstawie cech klinicznych
        i farmakoterapii. Badana jest także stabilność struktury podobieństw przy różnych metodach
        przetwarzania danych.
        """
    )

if menu == "PCA":
    st.subheader("Analiza PCA")

    pca_coords_path = TABLES / "pca_coords.csv"
    pca_plot = PLOTS / "pca_plot.png"
    pca_insulin = PLOTS / "pca_insulin.png"
    pca_loadings = PLOTS / "pca_loadings.png"

    if pca_coords_path.exists():
        pca_df = pd.read_csv(pca_coords_path)

        st.markdown("### Filtry")

        c1, c2 = st.columns(2)

        with c1:
            readmitted_filter = st.multiselect(
                "Readmitted",
                options=sorted(pca_df["readmitted"].dropna().unique().tolist()),
                default=sorted(pca_df["readmitted"].dropna().unique().tolist())
            )

        with c2:
            insulin_filter = st.multiselect(
                "Insulin",
                options=sorted(pca_df["insulin"].dropna().unique().tolist()),
                default=sorted(pca_df["insulin"].dropna().unique().tolist())
            )

        filtered_df = pca_df[
            pca_df["readmitted"].isin(readmitted_filter) &
            pca_df["insulin"].isin(insulin_filter)
        ]

        st.markdown(f"**Liczba punktów po filtrowaniu:** {len(filtered_df)}")

        st.markdown("### PCA filtrowane")
        st.scatter_chart(
            filtered_df.set_index("encounter_id")[["PC1", "PC2"]],
            use_container_width=True
        )

    else:
        st.warning("Brak pliku pca_coords.csv")

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**PCA pokolorowane readmission**")
        if pca_plot.exists():
            st.image(str(pca_plot), use_container_width=True)
        else:
            st.warning("Brak pliku pca_plot.png")

    with c2:
        st.markdown("**PCA pokolorowane insulinoterapią**")
        if pca_insulin.exists():
            st.image(str(pca_insulin), use_container_width=True)
        else:
            st.warning("Brak pliku pca_insulin.png")

    st.markdown("---")
    st.markdown("**Najważniejsze zmienne w PCA**")
    if pca_loadings.exists():
        st.image(str(pca_loadings), use_container_width=True)
    else:
        st.warning("Brak pliku pca_loadings.png")

if menu == "Terapia i populacja":
    st.subheader("Analizy populacyjne i farmakoterapeutyczne")

    c1, c2 = st.columns(2)

    med_plot = PLOTS / "medication_usage.png"
    age_meds_plot = PLOTS / "age_vs_medications.png"
    age_insulin_plot = PLOTS / "age_vs_insulin.png"

    with c1:
        st.markdown("**Najczęściej stosowane leki**")
        if med_plot.exists():
            st.image(str(med_plot), use_container_width=True)
        else:
            st.warning("Brak pliku medication_usage.png")

    with c2:
        st.markdown("**Średnia liczba leków a wiek**")
        if age_meds_plot.exists():
            st.image(str(age_meds_plot), use_container_width=True)
        else:
            st.warning("Brak pliku age_vs_medications.png")

    st.markdown("---")
    st.markdown("**Odsetek insulinoterapii a wiek**")
    if age_insulin_plot.exists():
        st.image(str(age_insulin_plot), use_container_width=True)
    else:
        st.warning("Brak pliku age_vs_insulin.png")

    st.info(
        "Ta sekcja pokazuje profil farmakoterapii oraz zależność intensywności leczenia od wieku pacjentów."
    )

if menu == "Stabilność":
    st.subheader("Stabilność struktury podobieństw")

    if stability_df is not None:
        st.dataframe(stability_df, use_container_width=True)
        st.success(
            "Współczynnik Jaccarda pokazuje, jak bardzo zmienia się lokalna struktura sąsiedztwa po zmianie sposobu normalizacji danych."
        )
    else:
        st.warning("Brak pliku stability_jaccard.csv")

    if loadings_path.exists():
        with st.expander("Tabela PCA loadings"):
            st.dataframe(pd.read_csv(loadings_path), use_container_width=True)

if menu == "Dane wejściowe":
    st.subheader("Statystyki danych wejściowych")

    if input_stats is not None:
        st.dataframe(input_stats, use_container_width=True)
    else:
        st.warning("Brak pliku input_stats.csv")

if menu == "Wnioski":
    st.subheader("Najważniejsze wnioski z analizy")

    st.success(
        "Projekt pokazuje, że podobieństwo hospitalizacji pacjentów z cukrzycą zależy zarówno od cech klinicznych, jak i od decyzji technicznych związanych z przetwarzaniem danych."
    )

    st.markdown(
        """
        ### 1. Struktura danych
        Analiza PCA pokazuje, że hospitalizacje pacjentów z cukrzycą tworzą wyraźną strukturę
        w przestrzeni cech klinicznych i farmakoterapeutycznych. Oznacza to, że przypadki nie są
        rozłożone losowo, lecz grupują się według podobnych profili leczenia i hospitalizacji.
        """
    )

    st.markdown(
        """
        ### 2. Najważniejsze cechy wpływające na podobieństwo
        Największy wpływ na strukturę danych mają:
        - liczba leków,
        - długość hospitalizacji,
        - liczba procedur laboratoryjnych,
        - liczba diagnoz,
        - cechy demograficzne pacjentów.
        """
    )

    st.markdown(
        """
        ### 3. Znaczenie insulinoterapii
        Kolorowanie PCA według insulinoterapii sugeruje, że intensywność leczenia insuliną
        częściowo wpływa na pozycję hospitalizacji w przestrzeni podobieństw. Jednocześnie
        nie obserwuje się całkowicie rozłącznych klastrów, co wskazuje na wieloczynnikowy
        charakter danych klinicznych.
        """
    )

    st.markdown(
        """
        ### 4. Wpływ wyboru metody przetwarzania
        Analiza stabilności z użyciem współczynnika Jaccarda pokazuje, że zmiana sposobu
        normalizacji danych wpływa na lokalną strukturę sąsiedztwa. Oznacza to, że decyzje
        techniczne nie są neutralne i mogą zmieniać wynik analizy podobieństwa pacjentów.
        """
    )

    st.markdown(
        """
        ### 5. Wnioski farmakoterapeutyczne
        Dodatkowe analizy populacyjne pokazują:
        - które leki są najczęściej stosowane,
        - jak zmienia się liczba leków wraz z wiekiem,
        - jak zmienia się odsetek insulinoterapii w różnych grupach wiekowych.
        """
    )

    st.markdown(
        """
        ### 6. Porównanie próby i pełnego zbioru
        Dashboard umożliwia porównanie wyników dla próby 5000 hospitalizacji oraz dla pełnego
        zbioru danych, co pozwala ocenić, jak wielkość danych wpływa na stabilność i strukturę wyników.
        """
    )