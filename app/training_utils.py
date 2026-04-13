from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERACTIVE_DIR = PROCESSED_DIR / "interactive_runs"
INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_FILE_MAP = {
    "15min_clean": PROCESSED_DIR / "features_15min_clean.csv",
    "15min_extended": PROCESSED_DIR / "features_15min_extended.csv",
    "hourly_clean": PROCESSED_DIR / "features_hourly_clean.csv",
    "hourly_extended": PROCESSED_DIR / "features_hourly_extended.csv",
}


def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)
    ) * 100


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "smape": float(smape(y_true, y_pred)),
    }


def validate_ranges(
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
):
    if train_start > train_end:
        raise ValueError("Train pradžia negali būti vėliau nei train pabaiga.")
    if test_start > test_end:
        raise ValueError("Test pradžia negali būti vėliau nei test pabaiga.")
    if train_end >= test_start:
        raise ValueError("Train pabaiga turi būti anksčiau nei test pradžia.")


def prepare_split(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
):
    train_start_ts = pd.to_datetime(train_start)
    train_end_ts = pd.to_datetime(train_end)
    test_start_ts = pd.to_datetime(test_start)
    test_end_ts = pd.to_datetime(test_end)

    validate_ranges(train_start_ts, train_end_ts, test_start_ts, test_end_ts)

    train_df = df[(df["datetime"] >= train_start_ts) & (df["datetime"] <= train_end_ts)].copy()
    test_df = df[(df["datetime"] >= test_start_ts) & (df["datetime"] <= test_end_ts)].copy()

    if train_df.empty:
        raise ValueError("Train intervale nėra duomenų.")
    if test_df.empty:
        raise ValueError("Test intervale nėra duomenų.")

    return train_df, test_df


def make_run_dir(run_name: str) -> Path:
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_name.strip())
    if not safe_name:
        raise ValueError("run_name negali būti tuščias.")
    run_dir = INTERACTIVE_DIR / safe_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_features(dataset_name: str) -> pd.DataFrame:
    if dataset_name not in FEATURE_FILE_MAP:
        raise ValueError(f"Nežinomas dataset_name: {dataset_name}")

    path = FEATURE_FILE_MAP[dataset_name]
    if not path.exists():
        raise FileNotFoundError(f"Nerastas features failas: {path}")

    df = pd.read_csv(path)
    if "datetime" not in df.columns or "price" not in df.columns:
        raise ValueError(f"Faile {path} trūksta datetime arba price stulpelio.")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = df.dropna(subset=["datetime", "price"]).sort_values("datetime").reset_index(drop=True)
    return df


def train_xgboost_interactive(
    dataset_name: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    run_name: str,
) -> dict:
    df = load_features(dataset_name)
    train_df, test_df = prepare_split(df, train_start, train_end, test_start, test_end)
    run_dir = make_run_dir(run_name)

    X_train = train_df.drop(columns=["datetime", "price"])
    y_train = train_df["price"]

    X_test = test_df.drop(columns=["datetime", "price"])
    y_test = test_df["price"]

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)

    pred_df = test_df[["datetime", "price"]].copy()
    pred_df["predicted_price"] = y_pred
    pred_df["abs_error"] = (pred_df["predicted_price"] - pred_df["price"]).abs()
    pred_df["error"] = pred_df["predicted_price"] - pred_df["price"]

    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance_gain": model.feature_importances_,
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)

    top_errors_df = pred_df.sort_values("abs_error", ascending=False).head(30).reset_index(drop=True)

    pred_path = run_dir / f"xgb_predictions_{dataset_name}.csv"
    model_path = run_dir / f"xgb_model_{dataset_name}.pkl"
    importance_path = run_dir / f"xgb_feature_importance_{dataset_name}.csv"
    errors_path = run_dir / f"xgb_top_errors_{dataset_name}.csv"
    summary_path = run_dir / f"xgb_summary_{dataset_name}.json"

    pred_df.to_csv(pred_path, index=False)
    importance_df.to_csv(importance_path, index=False)
    top_errors_df.to_csv(errors_path, index=False)
    joblib.dump(model, model_path)

    summary = {
        "run_name": run_name,
        "model": "XGBoost",
        "dataset_name": dataset_name,
        "train_start": str(train_df["datetime"].min()),
        "train_end": str(train_df["datetime"].max()),
        "test_start": str(test_df["datetime"].min()),
        "test_end": str(test_df["datetime"].max()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(X_train.shape[1]),
        **metrics,
        "files": {
            "predictions": str(pred_path),
            "model": str(model_path),
            "importance": str(importance_path),
            "top_errors": str(errors_path),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "summary": summary,
        "predictions": pred_df,
        "importance": importance_df,
        "top_errors": top_errors_df,
        "run_dir": str(run_dir),
    }