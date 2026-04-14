from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from keras import layers
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
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

RANDOM_STATE = 42
TOP_N_ERRORS = 30


def set_seed(seed: int = RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


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


def make_prediction_df(test_df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    pred_df = test_df[["datetime", "price"]].copy()
    pred_df["actual_price"] = pred_df["price"]
    pred_df["predicted_price"] = y_pred
    pred_df["abs_error"] = (pred_df["predicted_price"] - pred_df["actual_price"]).abs()
    pred_df["error"] = pred_df["predicted_price"] - pred_df["actual_price"]
    return pred_df


def make_top_errors_df(pred_df: pd.DataFrame) -> pd.DataFrame:
    return pred_df.sort_values("abs_error", ascending=False).head(TOP_N_ERRORS).reset_index(drop=True)


def save_summary(summary_path: Path, summary: dict):
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


def save_common_outputs(
    run_dir: Path,
    model_label: str,
    dataset_name: str,
    pred_df: pd.DataFrame,
    top_errors_df: pd.DataFrame,
    summary: dict,
    importance_df: pd.DataFrame | None = None,
    model_obj=None,
    model_suffix: str = "pkl",
    extra_objects: dict[str, object] | None = None,
):
    prefix = model_label.lower()

    pred_path = run_dir / f"{prefix}_predictions_{dataset_name}.csv"
    errors_path = run_dir / f"{prefix}_top_errors_{dataset_name}.csv"
    summary_path = run_dir / f"{prefix}_summary_{dataset_name}.json"

    pred_df.to_csv(pred_path, index=False)
    top_errors_df.to_csv(errors_path, index=False)

    saved_files = {
        "predictions": str(pred_path),
        "top_errors": str(errors_path),
    }

    if importance_df is not None:
        importance_path = run_dir / f"{prefix}_feature_importance_{dataset_name}.csv"
        importance_df.to_csv(importance_path, index=False)
        saved_files["importance"] = str(importance_path)

    if model_obj is not None:
        model_path = run_dir / f"{prefix}_model_{dataset_name}.{model_suffix}"
        if model_suffix == "keras":
            model_obj.save(model_path)
        elif model_suffix == "cbm":
            model_obj.save_model(str(model_path))
        else:
            joblib.dump(model_obj, model_path)
        saved_files["model"] = str(model_path)

    if extra_objects:
        for name, obj in extra_objects.items():
            extra_path = run_dir / f"{prefix}_{name}_{dataset_name}.pkl"
            joblib.dump(obj, extra_path)
            saved_files[name] = str(extra_path)

    summary["files"] = saved_files
    save_summary(summary_path, summary)
    saved_files["summary"] = str(summary_path)

    return saved_files


def make_tree_importance_df(feature_names, values):
    return (
        pd.DataFrame({
            "feature": feature_names,
            "importance": values,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def get_profile(dataset_name: str) -> dict:
    is_15min = "15min" in dataset_name

    if is_15min:
        return {
            "is_15min": True,
            "valid_ratio": 0.10,
            "mlp_drop_cols": ["lag_1", "lag_2", "price_return_1"],
            "mlp_epochs": 120,
            "mlp_batch_size": 64,
            "mlp_lr": 3e-4,
            "lstm_seq_len": 32,
            "lstm_epochs": 80,
            "lstm_batch_size": 64,
            "lstm_lr": 3e-4,
        }

    return {
        "is_15min": False,
        "valid_ratio": 0.15,
        "mlp_drop_cols": [],
        "mlp_epochs": 120,
        "mlp_batch_size": 64,
        "mlp_lr": 5e-4,
        "lstm_seq_len": 24,
        "lstm_epochs": 80,
        "lstm_batch_size": 64,
        "lstm_lr": 3e-4,
    }


def prepare_scaled_tabular(train_df: pd.DataFrame, test_df: pd.DataFrame, drop_cols: list[str] | None = None):
    drop_cols = drop_cols or []

    X_train = train_df.drop(columns=["datetime", "price"])
    y_train = train_df["price"]

    X_test = test_df.drop(columns=["datetime", "price"])
    y_test = test_df["price"]

    existing_drop_cols = [c for c in drop_cols if c in X_train.columns]
    if existing_drop_cols:
        X_train = X_train.drop(columns=existing_drop_cols)
        X_test = X_test.drop(columns=existing_drop_cols)

    if X_train.shape[1] == 0:
        raise ValueError("Nėra feature stulpelių po filtravimo.")

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).flatten()

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train_scaled": y_train_scaled,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "dropped_cols": existing_drop_cols,
    }


def build_mlp(input_dim: int, is_15min: bool, learning_rate: float):
    if is_15min:
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.10),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.05),
            layers.Dense(16, activation="relu"),
            layers.Dense(1)
        ])
        loss_name = "mae"
    else:
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.15),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.10),
            layers.Dense(32, activation="relu"),
            layers.Dense(1)
        ])
        loss_name = "mse"

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_name,
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model


def create_sequences(X: np.ndarray, y: np.ndarray, datetimes: np.ndarray, seq_len: int):
    X_seq = []
    y_seq = []
    dt_seq = []

    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len:i])
        y_seq.append(y[i])
        dt_seq.append(datetimes[i])

    return (
        np.array(X_seq, dtype=np.float32),
        np.array(y_seq, dtype=np.float32),
        np.array(dt_seq),
    )


def build_lstm(sequence_length: int, n_features: int, learning_rate: float):
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.15),
        layers.LSTM(32),
        layers.Dropout(0.10),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mae",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model


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
    pred_df = make_prediction_df(test_df, y_pred)
    top_errors_df = make_top_errors_df(pred_df)

    importance_df = make_tree_importance_df(
        X_train.columns,
        model.feature_importances_,
    )

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
    }

    save_common_outputs(
        run_dir=run_dir,
        model_label="xgb",
        dataset_name=dataset_name,
        pred_df=pred_df,
        top_errors_df=top_errors_df,
        summary=summary,
        importance_df=importance_df,
        model_obj=model,
        model_suffix="pkl",
    )

    return {
        "summary": summary,
        "predictions": pred_df,
        "importance": importance_df,
        "top_errors": top_errors_df,
        "run_dir": str(run_dir),
    }


def train_lightgbm_interactive(
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

    model = LGBMRegressor(
        objective="regression",
        random_state=42,
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=8,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    pred_df = make_prediction_df(test_df, y_pred)
    top_errors_df = make_top_errors_df(pred_df)

    importance_df = make_tree_importance_df(
        X_train.columns,
        model.feature_importances_,
    )

    summary = {
        "run_name": run_name,
        "model": "LightGBM",
        "dataset_name": dataset_name,
        "train_start": str(train_df["datetime"].min()),
        "train_end": str(train_df["datetime"].max()),
        "test_start": str(test_df["datetime"].min()),
        "test_end": str(test_df["datetime"].max()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(X_train.shape[1]),
        **metrics,
    }

    save_common_outputs(
        run_dir=run_dir,
        model_label="lgbm",
        dataset_name=dataset_name,
        pred_df=pred_df,
        top_errors_df=top_errors_df,
        summary=summary,
        importance_df=importance_df,
        model_obj=model,
        model_suffix="pkl",
    )

    return {
        "summary": summary,
        "predictions": pred_df,
        "importance": importance_df,
        "top_errors": top_errors_df,
        "run_dir": str(run_dir),
    }


def train_catboost_interactive(
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

    model = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=42,
        iterations=500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5.0,
        subsample=0.8,
        verbose=False,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    pred_df = make_prediction_df(test_df, y_pred)
    top_errors_df = make_top_errors_df(pred_df)

    importance_df = make_tree_importance_df(
        X_train.columns,
        model.get_feature_importance(),
    )

    summary = {
        "run_name": run_name,
        "model": "CatBoost",
        "dataset_name": dataset_name,
        "train_start": str(train_df["datetime"].min()),
        "train_end": str(train_df["datetime"].max()),
        "test_start": str(test_df["datetime"].min()),
        "test_end": str(test_df["datetime"].max()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(X_train.shape[1]),
        **metrics,
    }

    save_common_outputs(
        run_dir=run_dir,
        model_label="catboost",
        dataset_name=dataset_name,
        pred_df=pred_df,
        top_errors_df=top_errors_df,
        summary=summary,
        importance_df=importance_df,
        model_obj=model,
        model_suffix="cbm",
    )

    return {
        "summary": summary,
        "predictions": pred_df,
        "importance": importance_df,
        "top_errors": top_errors_df,
        "run_dir": str(run_dir),
    }


def train_mlp_interactive(
    dataset_name: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    run_name: str,
) -> dict:
    set_seed()

    df = load_features(dataset_name)
    train_df, test_df = prepare_split(df, train_start, train_end, test_start, test_end)
    run_dir = make_run_dir(run_name)

    profile = get_profile(dataset_name)
    prepared = prepare_scaled_tabular(
        train_df=train_df,
        test_df=test_df,
        drop_cols=profile["mlp_drop_cols"],
    )

    model = build_mlp(
        input_dim=prepared["X_train_scaled"].shape[1],
        is_15min=profile["is_15min"],
        learning_rate=profile["mlp_lr"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=10,
            restore_best_weights=True,
            verbose=0,
        ),
    ]

    model.fit(
        prepared["X_train_scaled"],
        prepared["y_train_scaled"],
        epochs=profile["mlp_epochs"],
        batch_size=profile["mlp_batch_size"],
        verbose=0,
        callbacks=callbacks,
        shuffle=False,
    )

    y_pred_scaled = model.predict(prepared["X_test_scaled"], verbose=0).flatten()
    y_pred = prepared["y_scaler"].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    metrics = compute_metrics(prepared["y_test"], y_pred)
    pred_df = make_prediction_df(test_df, y_pred)
    top_errors_df = make_top_errors_df(pred_df)
    importance_df = pd.DataFrame(columns=["feature", "importance"])

    summary = {
        "run_name": run_name,
        "model": "MLP",
        "dataset_name": dataset_name,
        "train_start": str(train_df["datetime"].min()),
        "train_end": str(train_df["datetime"].max()),
        "test_start": str(test_df["datetime"].min()),
        "test_end": str(test_df["datetime"].max()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(prepared["X_train"].shape[1]),
        "dropped_columns": prepared["dropped_cols"],
        **metrics,
    }

    save_common_outputs(
        run_dir=run_dir,
        model_label="mlp",
        dataset_name=dataset_name,
        pred_df=pred_df,
        top_errors_df=top_errors_df,
        summary=summary,
        importance_df=importance_df,
        model_obj=model,
        model_suffix="keras",
        extra_objects={
            "x_scaler": prepared["x_scaler"],
            "y_scaler": prepared["y_scaler"],
        },
    )

    return {
        "summary": summary,
        "predictions": pred_df,
        "importance": importance_df,
        "top_errors": top_errors_df,
        "run_dir": str(run_dir),
    }


def train_lstm_interactive(
    dataset_name: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    run_name: str,
) -> dict:
    set_seed()

    df = load_features(dataset_name)
    train_df, test_df = prepare_split(df, train_start, train_end, test_start, test_end)
    run_dir = make_run_dir(run_name)

    profile = get_profile(dataset_name)

    X_train_df = train_df.drop(columns=["datetime", "price"])
    y_train = train_df["price"].to_numpy()
    dt_train = train_df["datetime"].to_numpy()

    X_test_df = test_df.drop(columns=["datetime", "price"])
    y_test = test_df["price"].to_numpy()
    dt_test = test_df["datetime"].to_numpy()

    if X_train_df.shape[1] == 0:
        raise ValueError("Nėra feature stulpelių LSTM modeliui.")

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train_df)
    X_test_scaled = x_scaler.transform(X_test_df)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    X_train_seq, y_train_seq, _ = create_sequences(
        X_train_scaled, y_train_scaled, dt_train, profile["lstm_seq_len"]
    )
    X_test_seq, y_test_seq, dt_test_seq = create_sequences(
        X_test_scaled, y_test_scaled, dt_test, profile["lstm_seq_len"]
    )

    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        raise ValueError("Nepakanka duomenų LSTM sekų sukūrimui.")

    model = build_lstm(
        sequence_length=profile["lstm_seq_len"],
        n_features=X_train_seq.shape[2],
        learning_rate=profile["lstm_lr"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=10,
            restore_best_weights=True,
            verbose=0,
        ),
    ]

    model.fit(
        X_train_seq,
        y_train_seq,
        epochs=profile["lstm_epochs"],
        batch_size=profile["lstm_batch_size"],
        verbose=0,
        callbacks=callbacks,
        shuffle=False,
    )

    y_pred_scaled = model.predict(X_test_seq, verbose=0).flatten()
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_real = y_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

    pred_df = pd.DataFrame({
        "datetime": pd.to_datetime(dt_test_seq),
        "price": y_test_real,
    })
    pred_df["actual_price"] = pred_df["price"]
    pred_df["predicted_price"] = y_pred
    pred_df["abs_error"] = (pred_df["predicted_price"] - pred_df["actual_price"]).abs()
    pred_df["error"] = pred_df["predicted_price"] - pred_df["actual_price"]

    metrics = compute_metrics(y_test_real, y_pred)
    top_errors_df = make_top_errors_df(pred_df)
    importance_df = pd.DataFrame(columns=["feature", "importance"])

    summary = {
        "run_name": run_name,
        "model": "LSTM",
        "dataset_name": dataset_name,
        "train_start": str(train_df["datetime"].min()),
        "train_end": str(train_df["datetime"].max()),
        "test_start": str(pred_df["datetime"].min()),
        "test_end": str(pred_df["datetime"].max()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(pred_df)),
        "feature_count": int(X_train_df.shape[1]),
        "sequence_length": int(profile["lstm_seq_len"]),
        **metrics,
    }

    save_common_outputs(
        run_dir=run_dir,
        model_label="lstm",
        dataset_name=dataset_name,
        pred_df=pred_df,
        top_errors_df=top_errors_df,
        summary=summary,
        importance_df=importance_df,
        model_obj=model,
        model_suffix="keras",
        extra_objects={
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
        },
    )

    return {
        "summary": summary,
        "predictions": pred_df,
        "importance": importance_df,
        "top_errors": top_errors_df,
        "run_dir": str(run_dir),
    }