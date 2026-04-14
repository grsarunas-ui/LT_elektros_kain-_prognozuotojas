import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


STEP = 0.05
TOP_N_ERRORS = 50

BASE_MODELS = {
    "XGBoost": "xgb_predictions_{dataset}.csv",
    "LightGBM": "lgbm_predictions_{dataset}.csv",
    "CatBoost": "catboost_predictions_{dataset}.csv",
    "MLP": "mlp_predictions_{dataset}.csv",
    "LSTM": "lstm_predictions_{dataset}.csv",
    "Ridge": "ridge_predictions_{dataset}.csv",
}


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


def make_output_paths(dataset_name: str):
    processed_dir = Path("data/processed")
    reports_dir = Path("reports")

    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return {
        "predictions": processed_dir / f"ensemble_predictions_{dataset_name}.csv",
        "summary_json": reports_dir / f"ensemble_summary_{dataset_name}.json",
        "weight_search_csv": reports_dir / f"ensemble_weight_search_{dataset_name}.csv",
        "top_errors_csv": reports_dir / f"ensemble_top_errors_{dataset_name}.csv",
    }


def load_prediction_file(path: Path, model_name: str):
    if not path.exists():
        print(f"⚠️ Nerastas failas {model_name}: {path}")
        return None

    df = pd.read_csv(path)

    required = {"datetime", "price", "predicted_price"}
    if not required.issubset(df.columns):
        print(f"⚠️ Blogas prediction formatas {model_name}: {path}")
        return None

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["predicted_price"] = pd.to_numeric(df["predicted_price"], errors="coerce")

    df = df.dropna(subset=["datetime", "price", "predicted_price"]).copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    return df[["datetime", "price", "predicted_price"]].rename(
        columns={"predicted_price": f"pred_{model_name}"}
    )


def load_available_predictions(dataset_name: str):
    loaded = []
    processed_dir = Path("data/processed")

    for model_name, pattern in BASE_MODELS.items():
        path = processed_dir / pattern.format(dataset=dataset_name)
        df = load_prediction_file(path, model_name)
        if df is not None:
            loaded.append((model_name, df))

    if not loaded:
        raise ValueError(f"Nerasta prediction failų datasetui: {dataset_name}")

    merged = None
    model_names = []

    for model_name, df in loaded:
        model_names.append(model_name)
        if merged is None:
            merged = df.copy()
        else:
            merged = merged.merge(df.drop(columns=["price"]), on="datetime", how="inner")

    if merged is None or merged.empty:
        raise ValueError(f"Nepavyko suderinti prediction failų datasetui: {dataset_name}")

    merged = merged.sort_values("datetime").reset_index(drop=True)
    return merged, model_names


def generate_weight_combinations(n_models: int, step: float = STEP):
    units = int(round(1.0 / step))
    for combo in itertools.product(range(units + 1), repeat=n_models):
        if sum(combo) == units:
            yield [c * step for c in combo]


def search_best_weights(df: pd.DataFrame, model_names: list[str], step: float = STEP):
    pred_cols = [f"pred_{m}" for m in model_names]
    y_true = df["price"].to_numpy()

    results = []

    print("\n=== ENSEMBLE WEIGHT SEARCH ===")
    print("Modeliai:", model_names)
    print("Step:", step)

    total_tested = 0
    best_row = None

    for weights in generate_weight_combinations(len(model_names), step=step):
        y_pred = np.zeros(len(df), dtype=float)
        for w, col in zip(weights, pred_cols):
            y_pred += w * df[col].to_numpy()

        metrics = evaluate_predictions(y_true, y_pred)

        row = {
            "weights": json.dumps({m: float(w) for m, w in zip(model_names, weights)}, ensure_ascii=False),
            **{f"w_{m}": float(w) for m, w in zip(model_names, weights)},
            **metrics,
        }
        results.append(row)
        total_tested += 1

        if best_row is None or row["mae"] < best_row["mae"] or (
            row["mae"] == best_row["mae"] and row["rmse"] < best_row["rmse"]
        ):
            best_row = row

    results_df = pd.DataFrame(results).sort_values(["mae", "rmse"]).reset_index(drop=True)

    print(f"Ištestuota kombinacijų: {total_tested}")
    print("\n=== BEST ENSEMBLE ===")
    print(results_df.head(10).to_string(index=False))

    best_weights = {m: float(results_df.iloc[0][f"w_{m}"]) for m in model_names}
    return best_weights, results_df


def build_ensemble_predictions(df: pd.DataFrame, model_names: list[str], weights: dict[str, float]):
    out = df[["datetime", "price"]].copy()
    out["predicted_price"] = 0.0

    for model_name in model_names:
        out["predicted_price"] += weights[model_name] * df[f"pred_{model_name}"]

    out["abs_error"] = np.abs(out["price"] - out["predicted_price"])
    out["error"] = out["predicted_price"] - out["price"]
    out["ape_pct"] = np.abs(out["error"]) / (np.abs(out["price"]) + 1e-6) * 100

    return out


def save_top_errors(pred_df: pd.DataFrame, out_path: Path):
    top_errors = (
        pred_df.sort_values("abs_error", ascending=False)
        .head(TOP_N_ERRORS)
        .reset_index(drop=True)
    )
    top_errors.to_csv(out_path, index=False)
    return top_errors


def train_ensemble(dataset_name: str):
    out_paths = make_output_paths(dataset_name)

    df, model_names = load_available_predictions(dataset_name)
    best_weights, search_df = search_best_weights(df, model_names, step=STEP)
    search_df.to_csv(out_paths["weight_search_csv"], index=False)

    ensemble_df = build_ensemble_predictions(df, model_names, best_weights)
    ensemble_df.to_csv(out_paths["predictions"], index=False)

    top_errors_df = save_top_errors(ensemble_df, out_paths["top_errors_csv"])
    metrics = evaluate_predictions(ensemble_df["price"], ensemble_df["predicted_price"])

    summary = {
        "model": "Ensemble",
        "dataset": dataset_name,
        "rows": int(len(ensemble_df)),
        "models_used": model_names,
        "weights": best_weights,
        **metrics,
        "top_10_weight_search": search_df.head(10).to_dict(orient="records"),
        "output_files": {k: str(v) for k, v in out_paths.items()},
    }

    with open(out_paths["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== FINAL ENSEMBLE RESULTS ===")
    print(f"MAE:   {metrics['mae']:.2f}")
    print(f"RMSE:  {metrics['rmse']:.2f}")
    print(f"R2:    {metrics['r2']:.4f}")
    print(f"sMAPE: {metrics['smape']:.2f}%")
    print("\n=== BEST WEIGHTS ===")
    print(best_weights)
    print("\n=== TOP ERRORS ===")
    print(top_errors_df.head(10).to_string(index=False))

    return {
        "model": "Ensemble",
        "data": dataset_name,
        **metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset pavadinimas, pvz. hourly_clean, 15min_clean"
    )
    args = parser.parse_args()

    train_ensemble(args.dataset)