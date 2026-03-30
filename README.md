# DB3 – Analiza podobieństwa hospitalizacji pacjentów z cukrzycą

## Opis danych wejściowych
- **Źródło:** Diabetes Hospital Dataset (UCI)
- **Format:** CSV → SQLite
- **Rozmiar:** ~101 766 rekordów hospitalizacji
- **Cechy:** dane kliniczne i farmakoterapeutyczne (czas hospitalizacji, liczba leków, liczba diagnoz, insulinoterapia i in.)
- **Typy:** dane mieszane – liczbowe i kategoryczne

## Pipeline przetwarzania
1. **Wczytanie danych** – import CSV do bazy SQLite (`generate_db.py`)
2. **Statystyki wejściowe** – liczność, min/max, średnia, mediana, braki (`stats.py`)
3. **Feature engineering** – budowa macierzy cech pacjentów (`build_features.py`)
4. **Normalizacja** – z-score lub min-max (`normalize.py`)
5. **Redukcja wymiarów** – PCA 10D przed kNN (`pipeline.py`)
6. **Analiza podobieństwa** – k-nearest neighbors z BallTree (`similarity.py`)
7. **Wizualizacja** – PCA 2D, wykresy farmakoterapii (`pca_analysis.py`, `plots.py`)
8. **Stabilność** – porównanie sąsiedztwa kNN dla z-score vs min-max (Jaccard) (`stability.py`)

## Parametry
| Parametr | Wartość domyślna | Opis |
|---|---|---|
| `TOP_K_NEIGHBORS` | 10 | liczba sąsiadów w kNN |
| `RANDOM_SEED` | 42 | ziarno losowości |
| normalizacja | z-score / min-max | porównywane dwa warianty |
| PCA przed kNN | 10 komponentów | redukcja wymiarów |
| sample_size | 5000 / 20000 / None | rozmiar zbioru danych |

## Analiza wrażliwości
Porównano dwa warianty normalizacji (z-score vs min-max) przy użyciu współczynnika Jaccarda:
- sample (5 000 rekordów): Jaccard = 0.385
- medium (20 000 rekordów): Jaccard = 0.312

## Wyniki
Wyniki znajdują się w katalogu `results/`:
- `results/sample/tables/` – statystyki, macierz cech, sąsiedzi kNN, PCA, stabilność
- `results/sample/plots/` – wykresy PCA, leki, wiek vs leki, wiek vs insulina
- `results/medium/tables/` – analogicznie dla 20 000 rekordów
- `results/full/tables/` – analogicznie dla pełnego zbioru

## Jak uruchomić

### Instalacja zależności
```bash
pip install -r requirements.txt
```

### Uruchomienie pipeline (terminal)
```bash
# próbka 5000 rekordów
python main.py --sample 5000

# próbka 20000 rekordów
python main.py --sample 20000

# pełny zbiór
python main.py
```

### Uruchomienie dashboardu
```bash
streamlit run app.py
```

## Struktura repozytorium
```
project-db3-patients/
├── data/
│   ├── raw/                  # dane wejściowe (diabetic_data.csv)
│   └── processed/            # baza SQLite
├── src/
│   ├── pipeline.py           # główny pipeline
│   ├── generate_db.py        # import CSV → SQLite
│   ├── load_data.py          # wczytanie danych
│   ├── build_features.py     # feature engineering
│   ├── normalize.py          # normalizacja
│   ├── similarity.py         # kNN
│   ├── pca_analysis.py       # PCA
│   ├── stability.py          # analiza stabilności (Jaccard)
│   ├── plots.py              # wizualizacje
│   ├── additional_analysis.py
│   └── config.py             # parametry
├── results/
│   ├── sample/               # wyniki dla 5 000 rekordów
│   ├── medium/               # wyniki dla 20 000 rekordów
│   └── full/                 # wyniki dla pełnego zbioru
├── app.py                    # dashboard Streamlit
├── main.py                   # uruchomienie pipeline
└── requirements.txt
```
