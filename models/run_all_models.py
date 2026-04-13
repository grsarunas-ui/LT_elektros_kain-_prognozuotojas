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
TRAIN_LSTM = BASE_PATH / "models/train_lstm.py"

RESULTS_OUT = BASE_PATH / "data/processed/model_comparison.csv"


def run_model(script_path: Path, features_path: Path) -> bool:
    print("\n==============================")
    print(f"Running: {script_path.name} on {features_path.name}")

    result = subprocess.run(
        [
            PYTHON,
            str(script_path),
            "--features",
            str(features_path),
        ],
        capture_output=True,
        text=True
    )

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        if result.stderr:
            print(result.stderr)
        print(f"❌ Klaida paleidžiant {script_path.name} su {features_path.name}")
        return False

    print(f"✅ Baigta: {script_path.name} su {features_path.name}")
    return True


def parse_metrics(predictions_path: Path, model_name: str, data_name: str):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    if not predictions_path.exists():
        print(f"⚠️ Nerastas prognozių failas: {predictions_path}")
        return None

    df = pd.read_csv(predictions_path)

    required_cols = {"price", "predicted_price"}
    if not required_cols.issubset(df.columns):
        print(f"⚠️ Trūksta stulpelių faile: {predictions_path}")
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


def collect_model_metrics(dataset_name: str, model_name: str):
    prediction_map = {
        "XGBoost": BASE_PATH / f"data/processed/xgb_predictions_{dataset_name}.csv",
        "MLP": BASE_PATH / f"data/processed/mlp_predictions_{dataset_name}.csv",
        "LSTM": BASE_PATH / f"data/processed/lstm_predictions_{dataset_name}.csv",
    }

    if model_name not in prediction_map:
        raise ValueError(f"Nepalaikomas modelis: {model_name}")

    return parse_metrics(prediction_map[model_name], model_name, dataset_name)


def main():
    results = []

    model_scripts = [
        ("XGBoost", TRAIN_XGB),
        ("MLP", TRAIN_MLP),
        ("LSTM", TRAIN_LSTM),
    ]

    for features in FEATURE_FILES:
        print("\n" + "=" * 80)
        print(f"DATASET: {features.name}")

        if not features.exists():
            print(f"❌ Nerastas failas: {features}")
            continue

        dataset_name = features.stem.replace("features_", "")
        run_status = {}

        for model_name, script_path in model_scripts:
            if not script_path.exists():
                print(f"❌ Nerastas modelio skriptas: {script_path}")
                run_status[model_name] = False
                continue

            ok = run_model(script_path, features)
            run_status[model_name] = ok

        for model_name, _ in model_scripts:
            if not run_status.get(model_name, False):
                continue

            metrics = collect_model_metrics(dataset_name, model_name)
            if metrics:
                results.append(metrics)

    if results:
        df_results = (
            pd.DataFrame(results)
            .sort_values(["data", "mae"])
            .reset_index(drop=True)
        )

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