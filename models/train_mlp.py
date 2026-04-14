import argparse
import json
import random
from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


TEST_DAYS = 14
RANDOM_STATE = 42
TOP_N_ERRORS = 50

DEFAULT_VALID_RATIO = 0.15
DEFAULT_EPOCHS = 120
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 5e-4

VALID_RATIO_15MIN = 0.10
EPOCHS_15MIN = 120
BATCH_SIZE_15MIN = 64
LEARNING_RATE_15MIN = 3e-4

VALID_RATIO_HOURLY = 0.15
EPOCHS_HOURLY = 120
BATCH_SIZE_HOURLY = 64
LEARNING_RATE_HOURLY = 5e-4

MLP_15MIN_DROP_COLS = ["lag_1", "lag_2", "price_return_1"]


def set_seed(seed: int = RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


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


def get_train_valid_split(train_df: pd.DataFrame, valid_ratio: float):
    n_rows = len(train_df)
    n_valid = max(1, int(n_rows * valid_ratio))

    if n_rows - n_valid < 50:
        raise ValueError("Per mažai train duomenų validacijos išskyrimui.")

    split_idx = n_rows - n_valid

    train_fit = train_df.iloc[:split_idx].copy()
    valid = train_df.iloc[split_idx:].copy()

    return train_fit, valid


def make_output_paths(dataset_name: str):
    processed_dir = Path("data/processed")
    models_dir = Path("models")
    reports_dir = Path("reports")

    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return {
        "predictions": processed_dir / f"mlp_predictions_{dataset_name}.csv",
        "model": models_dir / f"mlp_model_{dataset_name}.keras",
        "x_scaler": models_dir / f"mlp_scaler_{dataset_name}.pkl",
        "y_scaler": models_dir / f"mlp_y_scaler_{dataset_name}.pkl",
        "history_csv": reports_dir / f"mlp_history_{dataset_name}.csv",
        "top_errors_csv": reports_dir / f"mlp_top_errors_{dataset_name}.csv",
        "summary_json": reports_dir / f"mlp_summary_{dataset_name}.json",
    }


def infer_dataset_profile(dataset_name: str) -> dict:
    is_15min = "15min" in dataset_name
    is_hourly = "hourly" in dataset_name

    if is_15min:
        return {
            "name": "15min",
            "valid_ratio": VALID_RATIO_15MIN,
            "epochs": EPOCHS_15MIN,
            "batch_size": BATCH_SIZE_15MIN,
            "learning_rate": LEARNING_RATE_15MIN,
            "drop_cols": MLP_15MIN_DROP_COLS,
            "loss": "mae",
        }

    if is_hourly:
        return {
            "name": "hourly",
            "valid_ratio": VALID_RATIO_HOURLY,
            "epochs": EPOCHS_HOURLY,
            "batch_size": BATCH_SIZE_HOURLY,
            "learning_rate": LEARNING_RATE_HOURLY,
            "drop_cols": [],
            "loss": "mse",
        }

    return {
        "name": "default",
        "valid_ratio": DEFAULT_VALID_RATIO,
        "epochs": DEFAULT_EPOCHS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "drop_cols": [],
        "loss": "mse",
    }


def build_mlp_15min(input_dim: int, learning_rate: float):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.10),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.05),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mae",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model


def build_mlp_hourly(input_dim: int, learning_rate: float):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.15),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.10),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model


def build_model_for_profile(input_dim: int, profile: dict):
    if profile["name"] == "15min":
        return build_mlp_15min(input_dim, profile["learning_rate"])
    return build_mlp_hourly(input_dim, profile["learning_rate"])


def drop_columns_for_profile(
    X_train_fit: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    drop_cols: list[str],
):
    cols_to_drop = [c for c in drop_cols if c in X_train_fit.columns]

    if cols_to_drop:
        X_train_fit = X_train_fit.drop(columns=cols_to_drop)
        X_valid = X_valid.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop)

    return X_train_fit, X_valid, X_test, cols_to_drop


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


def train_mlp(
    features_path: str,
    train_start: str | None = None,
    train_end: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
):
    set_seed()

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
    profile = infer_dataset_profile(dataset_name)
    out_paths = make_output_paths(dataset_name)

    print("\n==============================")
    print(f"DATA: {features_path}")
    print("Dataset:", dataset_name)
    print("Profile:", profile["name"])
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

    train_fit, valid = get_train_valid_split(train, valid_ratio=profile["valid_ratio"])

    X_train_fit = train_fit.drop(columns=["datetime", "price"])
    y_train_fit = train_fit["price"]

    X_valid = valid.drop(columns=["datetime", "price"])
    y_valid = valid["price"]

    X_test = test.drop(columns=["datetime", "price"])
    y_test = test["price"]

    X_train_fit, X_valid, X_test, dropped_cols = drop_columns_for_profile(
        X_train_fit, X_valid, X_test, profile["drop_cols"]
    )

    if X_train_fit.shape[1] == 0:
        raise ValueError("Nėra feature stulpelių po 'datetime' ir 'price' pašalinimo.")

    print("\nSplit summary:")
    print("Valid ratio:", profile["valid_ratio"])
    print("Train shape:", train.shape)
    print("Train fit shape:", train_fit.shape)
    print("Valid shape:", valid.shape)
    print("Test shape:", test.shape)
    print("Train range:", train["datetime"].min(), "->", train["datetime"].max())
    print("Valid range:", valid["datetime"].min(), "->", valid["datetime"].max())
    print("Test range:", test["datetime"].min(), "->", test["datetime"].max())
    print("Dropped columns for MLP:", dropped_cols if dropped_cols else "none")
    print("Feature count after filtering:", X_train_fit.shape[1])
    print("Loss:", profile["loss"])
    print("Epochs:", profile["epochs"])
    print("Batch size:", profile["batch_size"])
    print("Learning rate:", profile["learning_rate"])

    x_scaler = StandardScaler()
    X_train_fit_scaled = x_scaler.fit_transform(X_train_fit)
    X_valid_scaled = x_scaler.transform(X_valid)
    X_test_scaled = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_fit_scaled = y_scaler.fit_transform(
        y_train_fit.to_numpy().reshape(-1, 1)
    ).flatten()
    y_valid_scaled = y_scaler.transform(
        y_valid.to_numpy().reshape(-1, 1)
    ).flatten()

    model = build_model_for_profile(X_train_fit_scaled.shape[1], profile)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        ),
    ]

    print("\n=== TRAINING MLP ===")
    history = model.fit(
        X_train_fit_scaled,
        y_train_fit_scaled,
        validation_data=(X_valid_scaled, y_valid_scaled),
        epochs=profile["epochs"],
        batch_size=profile["batch_size"],
        verbose=1,
        callbacks=callbacks,
        shuffle=False
    )

    y_pred_scaled = model.predict(X_test_scaled, verbose=0).flatten()
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    metrics = evaluate_predictions(y_test, y_pred)

    print("\n=== MLP RESULTS ===")
    print(f"MAE:   {metrics['mae']:.2f}")
    print(f"RMSE:  {metrics['rmse']:.2f}")
    print(f"R2:    {metrics['r2']:.4f}")
    print(f"sMAPE: {metrics['smape']:.2f}%")

    predictions_df, top_errors_df = save_top_errors(
        test, y_pred, out_paths["top_errors_csv"]
    )
    predictions_df.to_csv(out_paths["predictions"], index=False)

    model.save(out_paths["model"])
    joblib.dump(x_scaler, out_paths["x_scaler"])
    joblib.dump(y_scaler, out_paths["y_scaler"])

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(out_paths["history_csv"], index=False)

    summary = {
        "model": "MLP",
        "dataset": dataset_name,
        "profile": profile["name"],
        "features_path": str(features_path),
        "row_count": int(len(df)),
        "feature_count": int(X_train_fit.shape[1]),
        "dropped_columns": dropped_cols,
        "train_rows": int(len(train)),
        "train_fit_rows": int(len(train_fit)),
        "valid_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "train_start": str(train["datetime"].min()),
        "train_end": str(train["datetime"].max()),
        "valid_start": str(valid["datetime"].min()),
        "valid_end": str(valid["datetime"].max()),
        "test_start": str(test["datetime"].min()),
        "test_end": str(test["datetime"].max()),
        "manual_split": all(v is not None for v in [train_start, train_end, test_start, test_end]),
        "epochs_trained": int(len(history.history["loss"])),
        "best_val_loss": float(np.min(history.history["val_loss"])),
        "loss_name": profile["loss"],
        "learning_rate": profile["learning_rate"],
        "batch_size": profile["batch_size"],
        **metrics,
        "output_files": {k: str(v) for k, v in out_paths.items()},
    }

    with open(out_paths["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"Išsaugota: {out_paths['predictions']}")
    print(f"Išsaugota: {out_paths['model']}")
    print(f"Išsaugota: {out_paths['x_scaler']}")
    print(f"Išsaugota: {out_paths['y_scaler']}")
    print(f"Išsaugota: {out_paths['history_csv']}")
    print(f"Išsaugota: {out_paths['top_errors_csv']}")
    print(f"Išsaugota: {out_paths['summary_json']}")

    return {
        "model": "MLP",
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

    train_mlp(
        features_path=args.features,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )