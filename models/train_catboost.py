import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


TEST_DAYS = 14
TOP_N_IMPORTANCE = 20
TOP_N_ERRORS = 50
N_SPLITS = 3
RANDOM_STATE = 42
FINAL_VALID_RATIO = 0.1

CANDIDATE_PARAMS = [
    {
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "subsample": 0.8,
    },
    {
        "iterations": 500,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 5.0,
        "subsample": 0.8,
    },
    {
        "iterations": 500,
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 5.0,
        "subsample": 0.9,
    },
    {
        "iterations": 700,
        "learning_rate": 0.02,
        "depth": 6,
        "l2_leaf_reg": 7.0,
        "subsample": 0.8,
    },
    {
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 3.0,
        "subsample": 0.9,
    },
    {
        "iterations": 500,
        "learning_rate": 0.03,
        "depth": 5,
        "l2_leaf_reg": 5.0,
        "subsample": 0.8,
    },
]


def smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)
    ) * 100


def evaluate_predictions(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "smape": float(smape(y_true, y_pred)),
    }


def normalize_catboost_params(params: dict) -> dict:
    params = dict(params)

    int_keys = ["iterations", "depth"]
    float_keys = ["learning_rate", "l2_leaf_reg", "subsample"]

    for key in int_keys:
        params[key] = int(params[key])

    for key in float_keys:
        params[key] = float(params[key])

    return params


def get_dynamic_split(df: pd.DataFrame, test_days: int = TEST_DAYS):
    max_dt = df["datetime"].max()
    test_start = max_dt - pd.Timedelta(days=test_days)

    train = df[df["datetime"] < test_start].copy()
    test = df[df["datetime"] >= test_start].copy()

    return train, test, test_start, max_dt


def get_explicit_split(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
):
    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    if train_start_ts > train_end_ts:
        raise ValueError("train_start negali būti vėliau nei train_end.")
    if test_start_ts > test_end_ts:
        raise ValueError("test_start negali būti vėliau nei test_end.")
    if train_end_ts >= test_start_ts:
        raise ValueError("train_end turi būti ankstesnė data nei test_start, kad nebūtų leakage.")

    train = df[
        (df["datetime"] >= train_start_ts) & (df["datetime"] <= train_end_ts)
    ].copy()
    test = df[
        (df["datetime"] >= test_start_ts) & (df["datetime"] <= test_end_ts)
    ].copy()

    return train, test, test_start_ts, test_end_ts


def resolve_split(
    df: pd.DataFrame,
    train_start: str | None = None,
    train_end: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
):
    provided = [train_start, train_end, test_start, test_end]
    if all(v is None for v in provided):
        return get_dynamic_split(df, test_days=TEST_DAYS)

    if any(v is None for v in provided):
        raise ValueError(
            "Jei nori rankinio split, privalai paduoti visas 4 datas: "
            "--train-start --train-end --test-start --test-end"
        )

    return get_explicit_split(df, train_start, train_end, test_start, test_end)


def make_output_paths(dataset_name: str):
    processed_dir = Path("data/processed")
    models_dir = Path("models")
    reports_dir = Path("reports")

    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return {
        "predictions": processed_dir / f"catboost_predictions_{dataset_name}.csv",
        "model": models_dir / f"catboost_model_{dataset_name}.cbm",
        "importance_csv": reports_dir / f"catboost_feature_importance_{dataset_name}.csv",
        "top_errors_csv": reports_dir / f"catboost_top_errors_{dataset_name}.csv",
        "summary_json": reports_dir / f"catboost_summary_{dataset_name}.json",
        "cv_results_csv": reports_dir / f"catboost_cv_results_{dataset_name}.csv",
    }


def save_feature_importance(model: CatBoostRegressor, feature_names, out_path: Path):
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.get_feature_importance()
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    importance_df.to_csv(out_path, index=False)
    return importance_df


def save_top_errors(test_df: pd.DataFrame, y_pred, out_path: Path):
    out = test_df[["datetime", "price"]].copy()
    out["predicted_price"] = y_pred
    out["abs_error"] = np.abs(out["price"] - out["predicted_price"])
    out["error"] = out["predicted_price"] - out["price"]
    out["ape_pct"] = np.abs(out["error"]) / (np.abs(out["price"]) + 1e-6) * 100

    top_errors = (
        out.sort_values("abs_error", ascending=False)
        .head(TOP_N_ERRORS)
        .reset_index(drop=True)
    )
    top_errors.to_csv(out_path, index=False)
    return out, top_errors


def make_final_train_valid_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    valid_ratio: float = FINAL_VALID_RATIO,
):
    n_rows = len(X_train)
    n_valid = max(1, int(n_rows * valid_ratio))

    if n_rows - n_valid < 20:
        raise ValueError("Per mažai train duomenų final validacijos išskyrimui.")

    split_idx = n_rows - n_valid

    X_fit = X_train.iloc[:split_idx].copy()
    y_fit = y_train.iloc[:split_idx].copy()

    X_valid = X_train.iloc[split_idx:].copy()
    y_valid = y_train.iloc[split_idx:].copy()

    return X_fit, y_fit, X_valid, y_valid


def tune_catboost_fast(X: pd.DataFrame, y: pd.Series):
    if len(X) < (N_SPLITS + 1):
        raise ValueError(
            f"Per mažai duomenų TimeSeriesSplit. Eilučių: {len(X)}, splitų: {N_SPLITS}"
        )

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    cv_rows = []

    print("\n=== FAST TIME SERIES CV (CATBOOST) ===")
    print(f"Kandidatų: {len(CANDIDATE_PARAMS)}")
    print(f"CV foldų: {N_SPLITS}")

    for i, params in enumerate(CANDIDATE_PARAMS, start=1):
        fold_mae = []
        fold_rmse = []
        fold_r2 = []
        fold_smape = []

        print("\n" + "=" * 70)
        print(f"[{i}/{len(CANDIDATE_PARAMS)}] Testing params:")
        print(params)

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

            print(
                f"\n--- Fold {fold_idx}/{N_SPLITS} | "
                f"train={X_train_fold.shape}, val={X_val_fold.shape} ---"
            )

            model = CatBoostRegressor(
                loss_function="MAE",
                eval_metric="MAE",
                random_seed=RANDOM_STATE,
                verbose=50,
                **params
            )

            model.fit(
                X_train_fold,
                y_train_fold,
                eval_set=(X_val_fold, y_val_fold),
                use_best_model=False
            )

            y_val_pred = model.predict(X_val_fold)
            metrics = evaluate_predictions(y_val_fold, y_val_pred)

            fold_mae.append(metrics["mae"])
            fold_rmse.append(metrics["rmse"])
            fold_r2.append(metrics["r2"])
            fold_smape.append(metrics["smape"])

            print(
                f"Fold {fold_idx} done -> "
                f"MAE={metrics['mae']:.3f}, "
                f"RMSE={metrics['rmse']:.3f}, "
                f"R2={metrics['r2']:.4f}, "
                f"sMAPE={metrics['smape']:.2f}%"
            )

        row = {
            **params,
            "cv_mae_mean": float(np.mean(fold_mae)),
            "cv_mae_std": float(np.std(fold_mae)),
            "cv_rmse_mean": float(np.mean(fold_rmse)),
            "cv_r2_mean": float(np.mean(fold_r2)),
            "cv_smape_mean": float(np.mean(fold_smape)),
        }
        cv_rows.append(row)

        print(
            "\nCandidate summary -> "
            f"MAE={row['cv_mae_mean']:.3f}, "
            f"RMSE={row['cv_rmse_mean']:.3f}, "
            f"R2={row['cv_r2_mean']:.4f}, "
            f"sMAPE={row['cv_smape_mean']:.2f}%"
        )

    cv_results_df = pd.DataFrame(cv_rows).sort_values(
        ["cv_mae_mean", "cv_rmse_mean"]
    ).reset_index(drop=True)

    best_params = cv_results_df.iloc[0][list(CANDIDATE_PARAMS[0].keys())].to_dict()
    best_params = normalize_catboost_params(best_params)

    return best_params, cv_results_df


def train_final_model(X_train, y_train, best_params):
    print("\n=== TRAINING FINAL CATBOOST ===")
    print("Best params:", best_params)

    X_fit, y_fit, X_valid, y_valid = make_final_train_valid_split(X_train, y_train)

    print(f"Final train split -> fit={X_fit.shape}, valid={X_valid.shape}")

    model = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=RANDOM_STATE,
        verbose=50,
        **best_params
    )

    model.fit(
        X_fit,
        y_fit,
        eval_set=(X_valid, y_valid),
        use_best_model=False
    )
    return model


def train_catboost(
    features_path: str,
    train_start: str | None = None,
    train_end: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
):
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

    train, test, _, _ = resolve_split(
        df,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    if train.empty:
        raise ValueError("Train rinkinys tuščias. Patikrink train datas.")
    if test.empty:
        raise ValueError("Test rinkinys tuščias. Patikrink test datas.")

    print("\n=== SPLIT ===")
    print("Train:", train.shape)
    print("Test:", test.shape)
    print("Train range:", train["datetime"].min(), "->", train["datetime"].max())
    print("Test range:", test["datetime"].min(), "->", test["datetime"].max())

    X_train = train.drop(columns=["datetime", "price"])
    y_train = train["price"]

    X_test = test.drop(columns=["datetime", "price"])
    y_test = test["price"]

    feature_names = X_train.columns.tolist()

    if X_train.shape[1] == 0:
        raise ValueError("Nėra feature stulpelių po 'datetime' ir 'price' pašalinimo.")

    best_params, cv_results_df = tune_catboost_fast(X_train, y_train)
    cv_results_df.to_csv(out_paths["cv_results_csv"], index=False)

    print("\n=== BEST CV RESULT ===")
    print(cv_results_df.head(10).to_string(index=False))

    model = train_final_model(X_train, y_train, best_params)

    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)

    predictions_df, top_errors_df = save_top_errors(
        test, y_pred, out_paths["top_errors_csv"]
    )
    predictions_df.to_csv(out_paths["predictions"], index=False)

    importance_df = save_feature_importance(
        model, feature_names, out_paths["importance_csv"]
    )
    model.save_model(str(out_paths["model"]))

    summary = {
        "model": "CatBoost",
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
        "manual_split": all(v is not None for v in [train_start, train_end, test_start, test_end]),
        "cv_n_splits": N_SPLITS,
        "cv_candidates": len(CANDIDATE_PARAMS),
        "best_params": best_params,
        "best_cv_result": cv_results_df.iloc[0].to_dict(),
        **metrics,
        "top_5_features": importance_df.head(5).to_dict(orient="records"),
        "output_files": {k: str(v) for k, v in out_paths.items()},
    }

    with open(out_paths["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("\n=== FINAL TEST RESULTS ===")
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
    print("✔ CV results saved")

    return {
        "model": "CatBoost",
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
    parser.add_argument("--train-start", default=None, help="Pvz. 2025-10-01")
    parser.add_argument("--train-end", default=None, help="Pvz. 2026-02-29 23:59:59")
    parser.add_argument("--test-start", default=None, help="Pvz. 2026-03-01")
    parser.add_argument("--test-end", default=None, help="Pvz. 2026-03-15 23:59:59")
    args = parser.parse_args()

    train_catboost(
        features_path=args.features,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )