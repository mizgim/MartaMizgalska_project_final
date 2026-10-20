# DB3 – Patient Similarity Analysis in Diabetes Hospitalisations

## Project Overview

The aim of this project is to investigate the similarity of hospitalisations involving patients with diabetes, using clinical and pharmacotherapy-related data, and to assess how stable the resulting similarity structure is under different data-preprocessing methods.

The project follows a DB3 approach based on an aggregated feature matrix of hospitalisation records. It combines a modular data-processing pipeline, machine-learning methods, and interactive visualisation.

## Data

- **Dataset:** Diabetes 130-US Hospitals
- **Source:** UCI Machine Learning Repository
- **Period:** 1999–2008
- **Healthcare settings:** 130 hospitals in the United States
- **Number of records:** 101,766 hospitalisations
- **Number of engineered features:** 25
- **Data types:** Mixed numerical and categorical variables
- **Input format:** CSV, imported into SQLite
- **Notable missing values:** `max_glu_serum`: 96,420; `A1Cresult`: 84,748

Dataset page: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

Download `diabetic_data.csv` and place it in the `data/raw/` directory.

## Processing Pipeline

The project uses a modular pipeline. Each stage is parameterised, and intermediate outputs are saved in the `results/` directory.

1. **SQLite database creation** – `generate_db.py` creates `patients.db`.
2. **Data loading and initial statistics** – `load_data.py` and `stats.py` produce `input_stats.csv`.
3. **Feature engineering** – `build_features.py` creates `feature_matrix.csv`.
4. **Normalization** – `normalize.py` applies z-score or min-max normalization.
5. **Dimensionality reduction** – `pipeline.py` applies PCA, reducing the feature space from 25 features to 10 principal components.
6. **Similarity analysis** – `similarity.py` uses k-nearest neighbours (kNN) to identify the 10 nearest neighbours for each hospitalisation.
7. **Stability analysis** – `stability.py` compares neighbour sets using the Jaccard coefficient.
8. **Visualization** – `plots.py` and `app.py` provide plots and an interactive dashboard.

## Configuration

Key parameters are defined in `src/config.py`, allowing settings to be changed without modifying the analytical code.

| Parameter | Value | Description |
|---|---|---|
| `TOP_K_NEIGHBORS` | 10 | Number of nearest neighbours returned by kNN |
| `N_QUERY_PATIENTS` | 100 | Number of records sampled for stability evaluation |
| `RANDOM_SEED` | 42 | Random seed for reproducibility |
| `NORMALIZATIONS` | z-score, min-max | Normalization methods compared |
| `METRICS` | Euclidean, Manhattan | Distance metrics compared |
| PCA before kNN | 10 components | Dimensionality reduction |
| `sample_size` | 5,000 / 20,000 / full dataset | Dataset size used in an analysis run |

## Methodology

### Feature Engineering

A feature matrix was constructed from the source data. It includes clinical variables such as length of stay, number of medications, number of diagnoses, and number of procedures, as well as pharmacotherapy-related variables encoded using binary or one-hot encoding. The resulting matrix contains 25 features per hospitalisation record.

### Normalization

Two normalization approaches are compared: z-score and min-max normalization. These methods place variables with different scales on comparable scales, although they transform the data differently.

### Principal Component Analysis (PCA)

Before kNN analysis, PCA reduces the feature space from 25 features to 10 principal components. The project report states that these components explain approximately 85–90% of the total variance. PCA reduces dimensionality, but it may also discard information; the retained components should not be assumed to preserve every clinically relevant relationship.

### Similarity Analysis with k-Nearest Neighbours

The project uses k-nearest neighbours (kNN), with a BallTree-based approach via `sklearn.neighbors.NearestNeighbors`, to identify the 10 nearest neighbours of each hospitalisation record. The reported distance metrics are Euclidean and Manhattan.

### Stability Analysis

The stability of the neighbour structure is assessed using the Jaccard coefficient. For 100 records selected using `RANDOM_SEED=42`, the sets of 10 nearest neighbours obtained with z-score and min-max normalization are compared. A higher Jaccard value indicates greater overlap between the two neighbour sets.

## Results

### Neighbourhood Stability

The reported Jaccard results are:

| Dataset subset | Number of records | Distance metric | Jaccard coefficient |
|---|---:|---|---:|
| Sample | 5,000 | Euclidean | 0.3850 |
| Sample | 5,000 | Manhattan | 0.3581 |
| Medium | 20,000 | Euclidean | 0.3122 |
| Medium | 20,000 | Manhattan | 0.3142 |
| Full dataset | 101,766 | Not calculated | Not calculated |

Stability was not calculated for the full dataset because of computational-performance constraints.

The reported Jaccard values indicate moderate overlap between the neighbour sets obtained using the two normalization methods. For the Euclidean metric, the coefficient decreases from 0.3850 for the 5,000-record sample to 0.3122 for the 20,000-record subset. The Manhattan results show a similar pattern.

These results suggest that preprocessing choices can materially affect which hospitalisations are identified as nearest neighbours. The observed decrease across dataset sizes should be interpreted cautiously: it does not, by itself, prove that dataset size causes lower stability.

### PCA Structure and Insulin-Related Patterns

According to the project report, the two-dimensional PCA visualization shows one large, elliptical distribution rather than clearly separated clusters. The report also describes a gradient associated with insulin therapy: records without insulin are concentrated in one region of the plot, while records involving insulin therapy tend toward another.

These are exploratory visual observations. A two-dimensional PCA plot alone cannot establish that distinct clusters do not exist in the full feature space, nor does a visual association with insulin therapy establish a causal or diagnostic relationship.

### Pharmacotherapy-Related Observations

The project report identifies metformin as the most frequently used medication in the population and reports that the number of medications increases with age. It also reports a higher proportion of insulin therapy in younger age groups, which the report associates with type 1 diabetes, compared with older groups.

The age-related insulin pattern should be treated as an observation rather than a confirmed inference about diabetes type unless diabetes type has been directly and reliably identified in the analysed data.

## Key Findings

1. The PCA visualization is described as showing a continuous distribution without clearly separated clusters.
2. The report identifies a visual gradient associated with insulin therapy in PCA space.
3. Jaccard coefficients range from 0.3122 to 0.3850 in the reported experiments, indicating that neighbour sets are sensitive to normalization choices.
4. Euclidean and Manhattan distance metrics produce broadly similar Jaccard values in the reported experiments.
5. Stability was evaluated on 5,000- and 20,000-record subsets; it was not calculated for the full dataset.

## Limitations

- **Historical data:** The dataset covers hospitalisations from 1999 to 2008. Its findings may not reflect current clinical practice, prescribing patterns, or patient populations.
- **Hospitalisation-level records:** The analysis compares hospitalisation records. These should not automatically be treated as unique individuals unless patient identifiers and repeated admissions have been handled explicitly.
- **Exploratory similarity, not clinical validation:** kNN identifies records that are close under a chosen feature representation and distance metric. This does not establish clinical equivalence or demonstrate usefulness for clinical decision-making.
- **No causal drug-safety conclusions:** Medication-related variables alone do not establish adverse drug reactions or causal relationships between treatments and outcomes. This project does not estimate drug-related risks and is not a pharmacovigilance signal-detection study.
- **Preprocessing sensitivity:** The Jaccard coefficient measures overlap between neighbour sets under different normalization choices. It does not measure the clinical validity of those neighbours.
- **Dimensionality reduction:** PCA can reduce computational complexity but may discard information. The retained components are not guaranteed to preserve all clinically relevant relationships.
- **Subset-based stability evaluation:** Stability was not calculated for the full dataset, so the reported values should not be assumed to represent full-dataset stability.
- **Generalizability:** Findings from this dataset should not automatically be generalized to other hospitals, time periods, or patient populations.

## Results and Reproducibility

The project report describes a GitHub repository named `MartaMizgalska_project_final`, containing the source code, `requirements.txt`, and the `results/` directory. The Streamlit dashboard can be launched with:

```bash
streamlit run app.py
```

Install the project dependencies first:

```bash
pip install -r requirements.txt
```

Run the pipeline from PyCharm using `main.py` and the appropriate `--sample` argument:

- `--sample 5000` – analyse a sample of 5,000 records
- `--sample 20000` – analyse a subset of 20,000 records
- No `--sample` argument – analyse the full dataset

Results are saved in the `results/` directory.

## Repository Structure

```text
project-db3-patients/
├── data/
│   ├── raw/                  # input data (diabetic_data.csv)
│   └── processed/            # SQLite database
├── src/
│   ├── pipeline.py           # main pipeline
│   ├── generate_db.py        # CSV-to-SQLite import
│   ├── load_data.py          # data loading
│   ├── build_features.py     # feature engineering
│   ├── normalize.py          # normalization
│   ├── similarity.py         # kNN analysis
│   ├── stability.py          # Jaccard stability analysis
│   ├── plots.py              # visualizations
│   ├── additional_analysis.py
│   └── config.py             # configuration parameters
├── results/
│   ├── sample/               # results for 5,000 records
│   ├── medium/               # results for 20,000 records
│   ├── full/                 # results for the full dataset
│   └── comparison_stability.csv
├── app.py                    # Streamlit dashboard
├── main.py                   # pipeline entry point
└── requirements.txt
```

## Data Source

Diabetes 130-US Hospitals Dataset, UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
