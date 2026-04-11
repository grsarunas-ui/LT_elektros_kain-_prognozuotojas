import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


TEST_DAYS = 14
TOP_N_IMPORTANCE = 20
TOP_N_ERRORS = 50


def smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)
    ) * 100


def get_dynamic_split(df: pd.DataFrame, test_days: int = TEST_DAYS):
    max_dt = df["datetime"].max()
    test_start = max_dt - pd.Timedelta(days=test_days)

    train = df[df["datetime"] < test_start].copy()
    test = df[df["datetime"] >= test_start].copy()

    return train, test, test_start, max_dt


def make_output_paths(dataset_name: str):
    processed_dir = Path("data/processed")
    models_dir = Path("models")
    reports_dir = Path("reports")

    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return {
        "predictions": processed_dir / f"xgb_predictions_{dataset_name}.csv",
        "model": models_dir / f"xgb_model_{dataset_name}.pkl",
        "importance_csv": reports_dir / f"xgb_feature_importance_{dataset_name}.csv",
        "top_errors_csv": reports_dir / f"xgb_top_errors_{dataset_name}.csv",
        "summary_json": reports_dir / f"xgb_summary_{dataset_name}.json",
    }


def evaluate_predictions(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "smape": float(smape(y_true, y_pred)),
    }


def save_feature_importance(model: XGBRegressor, feature_names, out_path: Path):
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_gain": model.feature_importances_
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)

    importance_df.to_csv(out_path, index=False)
    return importance_df


def save_top_errors(test_df: pd.DataFrame, y_pred, out_path: Path):
    out = test_df[["datetime", "price"]].copy()
    out["predicted_price"] = y_pred
    out["abs_error"] = np.abs(out["price"] - out["predicted_price"])
    out["error"] = out["predicted_price"] - out["price"]
    out["ape_pct"] = np.abs(out["error"]) / (np.abs(out["price"]) + 1e-6) * 100

    top_errors = out.sort_values("abs_error", ascending=False).head(TOP_N_ERRORS).reset_index(drop=True)
    top_errors.to_csv(out_path, index=False)
    return out, top_errors


def train_xgb(features_path: str):
    features_path = Path(features_path)

    if not features_path.exists():
        raise FileNotFoundError(f"Nerastas features failas: {features_path}")

    df = pd.read_csv(features_path)

    if "datetime" not in df.columns or "price" not in df.columns:
        raise ValueError(f"Faile {features_path} trūksta 'datetime' arba 'price' stulpelio.")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df = (
        df.dropna(subset=["datetime", "price"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    dataset_name = features_path.stem.replace("features_", "")
    out_paths = make_output_paths(dataset_name)

    print("\n=== LOADING DATA ===")
    print("Features file:", features_path)
    print("Dataset:", dataset_name)
    print("Shape:", df.shape)
    print("Range:", df["datetime"].min(), "->", df["datetime"].max())

    train, test, test_start, test_end = get_dynamic_split(df, test_days=TEST_DAYS)

    if train.empty:
        raise ValueError("Train rinkinys tuščias. Per mažai istorinių duomenų.")
    if test.empty:
        raise ValueError("Test rinkinys tuščias. Per mažai naujausių duomenų.")

    print("\n=== SPLIT ===")
    print("Train:", train.shape)
    print("Test:", test.shape)
    print("Test range:", test["datetime"].min(), "->", test["datetime"].max())

    X_train = train.drop(columns=["datetime", "price"])
    y_train = train["price"]

    X_test = test.drop(columns=["datetime", "price"])
    y_test = test["price"]

    feature_names = X_train.columns.tolist()

    print("\n=== TRAINING XGBOOST ===")
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
        objective="reg:squarederror"
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate_predictions(y_test, y_pred)

    predictions_df, top_errors_df = save_top_errors(test, y_pred, out_paths["top_errors_csv"])
    predictions_df.to_csv(out_paths["predictions"], index=False)

    importance_df = save_feature_importance(model, feature_names, out_paths["importance_csv"])
    joblib.dump(model, out_paths["model"])

    summary = {
        "model": "XGBoost",
        "dataset": dataset_name,
        "features_path": str(features_path),
        "row_count": int(len(df)),
        "feature_count": int(X_train.shape[1]),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_start": str(train["datetime"].min()),
        "train_end": str(train["datetime"].max()),
        "test_start": str(test["datetime"].min()),
        "test_end": str(test["datetime"].max()),
        **metrics,
        "top_5_features": importance_df.head(5).to_dict(orient="records"),
        "output_files": {k: str(v) for k, v in out_paths.items()},
    }

    with open(out_paths["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== RESULTS ===")
    print(f"MAE:   {metrics['mae']:.2f}")
    print(f"RMSE:  {metrics['rmse']:.2f}")
    print(f"R2:    {metrics['r2']:.4f}")
    print(f"sMAPE: {metrics['smape']:.2f}%")

    print("\n=== TOP FEATURES ===")
    print(importance_df.head(TOP_N_IMPORTANCE).to_string(index=False))

    print("\n=== TOP ERRORS ===")
    print(top_errors_df.head(10).to_string(index=False))

    print("\n=== DONE ===")
    print("✔ model saved")
    print("✔ predictions saved")
    print("✔ feature importance saved")
    print("✔ errors saved")

    return {
        "model": "XGBoost",
        "data": dataset_name,
        **metrics,
        "test_start": str(test["datetime"].min()),
        "test_end": str(test["datetime"].max()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        required=True,
        help="Pilnas arba santykinis kelias iki features CSV"
    )
    args = parser.parse_args()

    train_xgb(args.features)