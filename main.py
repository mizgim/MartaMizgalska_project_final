import argparse
from src.pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DB3 – Pipeline analizy hospitalizacji")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Rozmiar próbki (np. 5000, 20000). Brak = pełny zbiór."
    )
    args = parser.parse_args()
    run_pipeline(sample_size=args.sample)