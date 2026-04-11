import subprocess
from pathlib import Path
import pandas as pd


BASE_PATH = Path(
    "/Users/sarunas/Documents/LT Elektros kainų prognozuotojas/"
    "LT_elektros_kain-_prognozuotojas"
)

PYTHON = "/usr/local/bin/python3"

FEATURE_FILES = [
    BASE_PATH / "data/processed/features_15min_clean.csv",
    BASE_PATH / "data/processed/features_15min_extended.csv",
    BASE_PATH / "data/processed/features_hourly_clean.csv",
    BASE_PATH / "data/processed/features_hourly_extended.csv",
]

TRAIN_XGB = BASE_PATH / "models/train_xgb.py"
TRAIN_MLP = BASE_PATH / "models/train_mlp.py"

RESULTS_OUT = BASE_PATH / "data/processed/model_comparison.csv"


def run_model(script_path: Path, features_path: Path):
    print("\n==============================")
    print(f"Running: {script_path.name} on {features_path.name}")

    result = subprocess.run(
        [
            PYTHON,
            str(script_path),
            "--features",
            str(features_path)
        ],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        print(f"❌ Klaida paleidžiant {script_path.name} su {features_path.name}")
        return False

    return True


def parse_metrics(predictions_path: Path, model_name: str, data_name: str):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    if not predictions_path.exists():
        return None

    df = pd.read_csv(predictions_path)
    if not {"price", "predicted_price"}.issubset(df.columns):
        return None

    y_true = df["price"]
    y_pred = df["predicted_price"]

    smape = np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)
    ) * 100

    return {
        "model": model_name,
        "data": data_name,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "smape": float(smape),
    }


def main():
    results = []

    for features in FEATURE_FILES:
        if not features.exists():
            print(f"Nerastas failas: {features}")
            continue

        dataset_name = features.stem.replace("features_", "")

        run_model(TRAIN_XGB, features)
        run_model(TRAIN_MLP, features)

        xgb_pred = BASE_PATH / f"data/processed/xgb_predictions_{dataset_name}.csv"
        mlp_pred = BASE_PATH / f"data/processed/mlp_predictions_{dataset_name}.csv"

        xgb_metrics = parse_metrics(xgb_pred, "XGBoost", dataset_name)
        mlp_metrics = parse_metrics(mlp_pred, "MLP", dataset_name)

        if xgb_metrics:
            results.append(xgb_metrics)
        if mlp_metrics:
            results.append(mlp_metrics)

    if results:
        df_results = pd.DataFrame(results).sort_values(["data", "mae"]).reset_index(drop=True)
        print("\n==============================")
        print("FINAL COMPARISON")
        print(df_results)

        RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(RESULTS_OUT, index=False)
        print(f"\nIšsaugota: {RESULTS_OUT}")
    else:
        print("Nepavyko surinkti rezultatų.")


if __name__ == "__main__":
    main()