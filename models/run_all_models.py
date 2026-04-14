import argparse
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
TRAIN_LGBM = BASE_PATH / "models/train_lgbm.py"
TRAIN_CATBOOST = BASE_PATH / "models/train_catboost.py"
TRAIN_MLP = BASE_PATH / "models/train_mlp.py"
TRAIN_LSTM = BASE_PATH / "models/train_lstm.py"
TRAIN_ENSEMBLE = BASE_PATH / "models/train_ensemble.py"

RESULTS_OUT = BASE_PATH / "data/processed/model_comparison.csv"


def build_date_args(
    train_start: str | None,
    train_end: str | None,
    test_start: str | None,
    test_end: str | None,
) -> list[str]:
    provided = [train_start, train_end, test_start, test_end]

    if all(v is None for v in provided):
        return []

    if any(v is None for v in provided):
        raise ValueError(
            "Jei nori rankinio split, paduok visas 4 datas: "
            "--train-start --train-end --test-start --test-end"
        )

    return [
        "--train-start", train_start,
        "--train-end", train_end,
        "--test-start", test_start,
        "--test-end", test_end,
    ]


def run_feature_model(
    script_path: Path,
    features_path: Path,
    train_start: str | None = None,
    train_end: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
) -> bool:
    print("\n==============================")
    print(f"Running: {script_path.name} on {features_path.name}")

    cmd = [
        PYTHON,
        str(script_path),
        "--features",
        str(features_path),
        *build_date_args(train_start, train_end, test_start, test_end),
    ]

    print("Command:", " ".join(cmd))

    result = subprocess.run(
        cmd,
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


def run_ensemble_model(script_path: Path, dataset_name: str) -> bool:
    print("\n==============================")
    print(f"Running: {script_path.name} on dataset={dataset_name}")

    result = subprocess.run(
        [
            PYTHON,
            str(script_path),
            "--dataset",
            dataset_name,
        ],
        capture_output=True,
        text=True
    )

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        if result.stderr:
            print(result.stderr)
        print(f"❌ Klaida paleidžiant {script_path.name} su dataset={dataset_name}")
        return False

    print(f"✅ Baigta: {script_path.name} su dataset={dataset_name}")
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

    y_true = pd.to_numeric(df["price"], errors="coerce")
    y_pred = pd.to_numeric(df["predicted_price"], errors="coerce")

    valid_mask = y_true.notna() & y_pred.notna()
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        print(f"⚠️ Nėra validžių eilučių metrikoms: {predictions_path}")
        return None

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
        "LightGBM": BASE_PATH / f"data/processed/lgbm_predictions_{dataset_name}.csv",
        "CatBoost": BASE_PATH / f"data/processed/catboost_predictions_{dataset_name}.csv",
        "MLP": BASE_PATH / f"data/processed/mlp_predictions_{dataset_name}.csv",
        "LSTM": BASE_PATH / f"data/processed/lstm_predictions_{dataset_name}.csv",
        "Ensemble": BASE_PATH / f"data/processed/ensemble_predictions_{dataset_name}.csv",
    }

    if model_name not in prediction_map:
        raise ValueError(f"Nepalaikomas modelis: {model_name}")

    return parse_metrics(prediction_map[model_name], model_name, dataset_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-start", default=None, help="Pvz. 2025-10-01")
    parser.add_argument("--train-end", default=None, help="Pvz. 2026-02-29 23:59:59")
    parser.add_argument("--test-start", default=None, help="Pvz. 2026-03-01")
    parser.add_argument("--test-end", default=None, help="Pvz. 2026-03-15 23:59:59")
    args = parser.parse_args()

    results = []

    base_model_scripts = [
        ("XGBoost", TRAIN_XGB),
        ("LightGBM", TRAIN_LGBM),
        ("CatBoost", TRAIN_CATBOOST),
        ("MLP", TRAIN_MLP),
        ("LSTM", TRAIN_LSTM),
    ]

    manual_split = all(
        v is not None for v in [args.train_start, args.train_end, args.test_start, args.test_end]
    )

    print("\n" + "=" * 80)
    print("RUN CONFIG")
    if manual_split:
        print(f"Train: {args.train_start} -> {args.train_end}")
        print(f"Test:  {args.test_start} -> {args.test_end}")
    else:
        print(f"Naudojamas default dynamic split: paskutinės {TEST_DAYS if 'TEST_DAYS' in globals() else 14} dienų testas")

    for features in FEATURE_FILES:
        print("\n" + "=" * 80)
        print(f"DATASET: {features.name}")

        if not features.exists():
            print(f"❌ Nerastas failas: {features}")
            continue

        dataset_name = features.stem.replace("features_", "")
        run_status = {}

        for model_name, script_path in base_model_scripts:
            if not script_path.exists():
                print(f"❌ Nerastas modelio skriptas: {script_path}")
                run_status[model_name] = False
                continue

            ok = run_feature_model(
                script_path=script_path,
                features_path=features,
                train_start=args.train_start,
                train_end=args.train_end,
                test_start=args.test_start,
                test_end=args.test_end,
            )
            run_status[model_name] = ok

        for model_name, _ in base_model_scripts:
            if not run_status.get(model_name, False):
                continue

            metrics = collect_model_metrics(dataset_name, model_name)
            if metrics:
                if manual_split:
                    metrics["train_start"] = args.train_start
                    metrics["train_end"] = args.train_end
                    metrics["test_start"] = args.test_start
                    metrics["test_end"] = args.test_end
                results.append(metrics)

        if TRAIN_ENSEMBLE.exists():
            ensemble_ok = run_ensemble_model(TRAIN_ENSEMBLE, dataset_name)
            if ensemble_ok:
                ensemble_metrics = collect_model_metrics(dataset_name, "Ensemble")
                if ensemble_metrics:
                    if manual_split:
                        ensemble_metrics["train_start"] = args.train_start
                        ensemble_metrics["train_end"] = args.train_end
                        ensemble_metrics["test_start"] = args.test_start
                        ensemble_metrics["test_end"] = args.test_end
                    results.append(ensemble_metrics)
        else:
            print(f"⚠️ Nerastas ensemble skriptas: {TRAIN_ENSEMBLE}")

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