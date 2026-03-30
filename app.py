import streamlit as st
import pandas as pd
import base64
from pathlib import Path
from src.pipeline import run_pipeline

st.set_page_config(
    page_title="Analiza hospitalizacji diabetologicznych",
    layout="wide"
)

BASE = Path(__file__).resolve().parent

def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_base64 = get_base64_image(BASE / "tabletkitlo1.png")

st.markdown(f"""
    <style>
    [data-testid="stSidebar"] {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        border-right: 2px solid rgba(200, 150, 150, 0.4);
    }}
    [data-testid="stSidebar"]::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(255, 255, 255, 0.75);
        z-index: 0;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        position: relative;
        z-index: 1;
    }}
    </style>
""", unsafe_allow_html=True)

dataset_type = st.sidebar.radio(
    "Wybierz dane",
    ["full", "medium", "sample"],
    key="dataset_radio"
)

menu = st.sidebar.radio(
    "Wybierz sekcję",
    ["Przegląd", "PCA", "Terapia i populacja", "Stabilność", "Dane wejściowe", "Wnioski"],
    key="menu_radio"
)

TABLES = BASE / "results" / dataset_type / "tables"
PLOTS = BASE / "results" / dataset_type / "plots"

st.title("Analiza hospitalizacji diabetologicznych")
st.caption("Interaktywny dashboard projektu rocznego — analiza podobieństwa hospitalizacji pacjentów z cukrzycą")

st.info(f"Aktualnie wybrany zbiór: {dataset_type}")

if st.button("Uruchom pipeline"):
    progress = st.progress(0, text="Inicjalizacja...")
    status = st.empty()

    def update_status(msg, pct):
        progress.progress(pct, text=msg)
        status.info(msg)

    if dataset_type == "sample":
        run_pipeline(sample_size=5000, status_callback=update_status)
    elif dataset_type == "medium":
        run_pipeline(sample_size=20000, status_callback=update_status)
    else:
        run_pipeline(sample_size=None, status_callback=update_status)

    progress.progress(100, text="Gotowe!")
    st.success("Pipeline zakończony.")
    st.rerun()

feature_matrix_path = TABLES / "feature_matrix.csv"
input_stats_path = TABLES / "input_stats.csv"
stability_path = TABLES / "stability_jaccard.csv"
loadings_path = TABLES / "pca_loadings.csv"

feature_matrix = pd.read_csv(feature_matrix_path) if feature_matrix_path.exists() else None
input_stats = pd.read_csv(input_stats_path) if input_stats_path.exists() else None
stability_df = pd.read_csv(stability_path) if stability_path.exists() else None

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
        jaccard_val = stability_df["jaccard_mean"].iloc[0]
        col3.metric("Średnia stabilność (Jaccard)",
                    round(float(jaccard_val), 3) if pd.notna(jaccard_val) else "n/d")
    else:
        col3.metric("Średnia stabilność (Jaccard)", "brak")

    st.markdown("---")
    st.markdown("""
    ### Cel projektu
    Projekt analizuje podobieństwo hospitalizacji pacjentów z cukrzycą na podstawie cech klinicznych
    i farmakoterapii. Badana jest także stabilność struktury podobieństw przy różnych metodach
    przetwarzania danych.
    """)

if menu == "PCA":
    st.subheader("Analiza PCA")
    pca_coords_path = TABLES / "pca_coords.csv"
    pca_plot = PLOTS / "pca_plot.png"
    pca_insulin = PLOTS / "pca_insulin.png"
    pca_loadings = PLOTS / "pca_loadings.png"

    if pca_coords_path.exists():
        import plotly.express as px

        pca_df = pd.read_csv(pca_coords_path)

        color_by = st.radio(
            "Koloruj według",
            ["insulin", "readmitted"],
            horizontal=True
        )

        if len(pca_df) > 5000:
            display_df = pca_df.sample(n=5000, random_state=42)
            st.caption(f"Wyświetlono 5000 z {len(pca_df)} punktów (próbka losowa)")
        else:
            display_df = pca_df

        fig = px.scatter(
            display_df,
            x="PC1",
            y="PC2",
            color=color_by,
            hover_data={"encounter_id": True, "PC1": ":.2f", "PC2": ":.2f"},
            title=f"PCA hospitalizacji — kolor: {color_by}",
            opacity=0.6,
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(
            legend_title_text=color_by,
            height=550,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Brak pliku pca_coords.csv – uruchom pipeline.")

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

    st.info("Ta sekcja pokazuje profil farmakoterapii oraz zależność intensywności leczenia od wieku pacjentów.")

if menu == "Stabilność":
    st.subheader("Stabilność struktury podobieństw")

    if stability_df is not None:
        st.markdown("**Wyniki dla aktualnie wybranego zbioru:**")
        st.dataframe(stability_df, use_container_width=True)
        st.success("Współczynnik Jaccarda pokazuje, jak bardzo zmienia się lokalna struktura sąsiedztwa po zmianie sposobu normalizacji danych.")
    else:
        st.warning("Brak pliku stability_jaccard.csv")

    st.markdown("---")
    st.markdown("**Porównanie wszystkich zbiorów danych:**")
    comparison_path = BASE / "results" / "comparison_stability.csv"
    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path)
        st.dataframe(comparison_df, use_container_width=True)
    else:
        st.warning("Brak tabeli porównawczej – uruchom pipeline dla sample i medium.")

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

    st.success("Projekt pokazuje, że podobieństwo hospitalizacji pacjentów z cukrzycą zależy zarówno od cech klinicznych, jak i od decyzji technicznych związanych z przetwarzaniem danych.")

    st.markdown("""
    ### 1. Struktura danych
    Hospitalizacje tworzą jedno duże, eliptyczne skupisko w przestrzeni PCA bez wyraźnie
    rozłącznych klastrów. Oznacza to, że przypadki różnią się od siebie stopniowo,
    a nie skokowo – nie istnieją ostro oddzielone typy hospitalizacji diabetologicznych.
    """)

    st.markdown("""
    ### 2. Najważniejsze cechy wpływające na podobieństwo
    Największy wpływ na strukturę danych mają liczba leków, długość hospitalizacji,
    liczba procedur laboratoryjnych, liczba diagnoz oraz cechy demograficzne pacjentów.
    """)

    st.markdown("""
    ### 3. Znaczenie insulinoterapii
    We wszystkich trzech zbiorach danych widoczny jest gradient insulinoterapii
    w przestrzeni PCA – pacjenci bez insuliny (No) koncentrują się w lewej górnej
    części wykresu, natomiast pacjenci z aktywną insulinoterapią (Steady, Up, Down)
    przesuwają się ku prawej dolnej części. Gradient jest najbardziej wyraźny
    w pełnym zbiorze 101 766 rekordów, co sugeruje że insulinoterapia jest
    istotnym czynnikiem różnicującym profile hospitalizacji.
    """)

    st.markdown("""
    ### 4. Wpływ wyboru metody przetwarzania
    Analiza stabilności wykazała, że wartość współczynnika Jaccarda wyniosła 0.385
    dla próbki 5 000 rekordów oraz 0.312 dla 20 000 rekordów. Oznacza to, że wraz
    ze wzrostem zbioru danych struktura sąsiedztwa staje się mniej stabilna przy
    zmianie normalizacji z z-score na min-max – większy zbiór ujawnia większą
    wrażliwość na wybór metody przetwarzania danych.
    """)

    st.markdown("""
    ### 5. Wnioski farmakoterapeutyczne
    Dodatkowe analizy populacyjne pokazują które leki są najczęściej stosowane,
    jak rośnie liczba leków wraz z wiekiem pacjenta oraz jak zmienia się odsetek
    insulinoterapii w różnych grupach wiekowych.
    """)

    st.markdown("""
    ### 6. Porównanie trzech zbiorów danych
    Struktura PCA jest jakościowo zbliżona dla wszystkich trzech rozmiarów danych,
    co potwierdza spójność wyników. Jednocześnie stabilność Jaccarda spada wraz
    ze wzrostem zbioru (0.385 → 0.312), co sugeruje że w większych zbiorach
    lokalne sąsiedztwa są bardziej zróżnicowane i wrażliwsze na decyzje
    dotyczące preprocessingu.
    """)