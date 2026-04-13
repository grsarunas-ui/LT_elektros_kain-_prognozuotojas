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

VALID_RATIO_15MIN = 0.10
VALID_RATIO_HOURLY = 0.15

SEQUENCE_LENGTH_15MIN = 32
SEQUENCE_LENGTH_HOURLY = 24

EPOCHS_15MIN = 80
EPOCHS_HOURLY = 80

BATCH_SIZE_15MIN = 64
BATCH_SIZE_HOURLY = 64

LEARNING_RATE_15MIN = 3e-4
LEARNING_RATE_HOURLY = 3e-4


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


def get_train_valid_split(train_df: pd.DataFrame, valid_ratio: float):
    n_rows = len(train_df)
    n_valid = max(1, int(n_rows * valid_ratio))

    if n_rows - n_valid < 100:
        raise ValueError("Per mažai train duomenų validacijos išskyrimui.")

    split_idx = n_rows - n_valid
    train_fit = train_df.iloc[:split_idx].copy()
    valid = train_df.iloc[split_idx:].copy()

    return train_fit, valid


def infer_dataset_profile(dataset_name: str) -> dict:
    if "15min" in dataset_name:
        return {
            "name": "15min",
            "valid_ratio": VALID_RATIO_15MIN,
            "sequence_length": SEQUENCE_LENGTH_15MIN,
            "epochs": EPOCHS_15MIN,
            "batch_size": BATCH_SIZE_15MIN,
            "learning_rate": LEARNING_RATE_15MIN,
        }

    return {
        "name": "hourly",
        "valid_ratio": VALID_RATIO_HOURLY,
        "sequence_length": SEQUENCE_LENGTH_HOURLY,
        "epochs": EPOCHS_HOURLY,
        "batch_size": BATCH_SIZE_HOURLY,
        "learning_rate": LEARNING_RATE_HOURLY,
    }


def make_output_paths(dataset_name: str):
    processed_dir = Path("data/processed")
    models_dir = Path("models")
    reports_dir = Path("reports")

    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return {
        "predictions": processed_dir / f"lstm_predictions_{dataset_name}.csv",
        "model": models_dir / f"lstm_model_{dataset_name}.keras",
        "x_scaler": models_dir / f"lstm_scaler_{dataset_name}.pkl",
        "y_scaler": models_dir / f"lstm_y_scaler_{dataset_name}.pkl",
        "history_csv": reports_dir / f"lstm_history_{dataset_name}.csv",
        "top_errors_csv": reports_dir / f"lstm_top_errors_{dataset_name}.csv",
        "summary_json": reports_dir / f"lstm_summary_{dataset_name}.json",
    }


def build_lstm_model(sequence_length: int, n_features: int, learning_rate: float):
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


def train_lstm(features_path: str):
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

    train, test, _, _ = get_dynamic_split(df, test_days=TEST_DAYS)

    if train.empty:
        raise ValueError("Train rinkinys tuščias.")
    if test.empty:
        raise ValueError("Test rinkinys tuščias.")

    train_fit, valid = get_train_valid_split(train, valid_ratio=profile["valid_ratio"])

    X_train_fit_df = train_fit.drop(columns=["datetime", "price"])
    y_train_fit = train_fit["price"].to_numpy()

    X_valid_df = valid.drop(columns=["datetime", "price"])
    y_valid = valid["price"].to_numpy()

    X_test_df = test.drop(columns=["datetime", "price"])
    y_test = test["price"].to_numpy()

    dt_train_fit = train_fit["datetime"].to_numpy()
    dt_valid = valid["datetime"].to_numpy()
    dt_test = test["datetime"].to_numpy()

    if X_train_fit_df.shape[1] == 0:
        raise ValueError("Nėra feature stulpelių po 'datetime' ir 'price' pašalinimo.")

    print("\nDynamic split:")
    print("Test days:", TEST_DAYS)
    print("Valid ratio:", profile["valid_ratio"])
    print("Sequence length:", profile["sequence_length"])
    print("Train shape:", train.shape)
    print("Train fit shape:", train_fit.shape)
    print("Valid shape:", valid.shape)
    print("Test shape:", test.shape)
    print("Feature count:", X_train_fit_df.shape[1])

    x_scaler = StandardScaler()
    X_train_fit_scaled = x_scaler.fit_transform(X_train_fit_df)
    X_valid_scaled = x_scaler.transform(X_valid_df)
    X_test_scaled = x_scaler.transform(X_test_df)

    y_scaler = StandardScaler()
    y_train_fit_scaled = y_scaler.fit_transform(y_train_fit.reshape(-1, 1)).flatten()
    y_valid_scaled = y_scaler.transform(y_valid.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    X_train_seq, y_train_seq, dt_train_seq = create_sequences(
        X_train_fit_scaled, y_train_fit_scaled, dt_train_fit, profile["sequence_length"]
    )
    X_valid_seq, y_valid_seq, dt_valid_seq = create_sequences(
        X_valid_scaled, y_valid_scaled, dt_valid, profile["sequence_length"]
    )
    X_test_seq, y_test_seq, dt_test_seq = create_sequences(
        X_test_scaled, y_test_scaled, dt_test, profile["sequence_length"]
    )

    if len(X_train_seq) == 0 or len(X_valid_seq) == 0 or len(X_test_seq) == 0:
        raise ValueError("Nepakanka duomenų sekų sukūrimui. Mažink sequence length.")

    print("\nSequence shapes:")
    print("Train seq:", X_train_seq.shape)
    print("Valid seq:", X_valid_seq.shape)
    print("Test seq:", X_test_seq.shape)

    model = build_lstm_model(
        sequence_length=profile["sequence_length"],
        n_features=X_train_seq.shape[2],
        learning_rate=profile["learning_rate"]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1
        ),
    ]

    print("\n=== TRAINING LSTM ===")
    history = model.fit(
        X_train_seq,
        y_train_seq,
        validation_data=(X_valid_seq, y_valid_seq),
        epochs=profile["epochs"],
        batch_size=profile["batch_size"],
        verbose=1,
        callbacks=callbacks,
        shuffle=False
    )

    y_pred_scaled = model.predict(X_test_seq, verbose=0).flatten()
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_real = y_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

    test_results_df = pd.DataFrame({
        "datetime": pd.to_datetime(dt_test_seq),
        "price": y_test_real,
    })

    metrics = evaluate_predictions(y_test_real, y_pred)

    print("\n=== LSTM RESULTS ===")
    print(f"MAE:   {metrics['mae']:.2f}")
    print(f"RMSE:  {metrics['rmse']:.2f}")
    print(f"R2:    {metrics['r2']:.4f}")
    print(f"sMAPE: {metrics['smape']:.2f}%")

    predictions_df, top_errors_df = save_top_errors(
        test_results_df, y_pred, out_paths["top_errors_csv"]
    )
    predictions_df.to_csv(out_paths["predictions"], index=False)

    model.save(out_paths["model"])
    joblib.dump(x_scaler, out_paths["x_scaler"])
    joblib.dump(y_scaler, out_paths["y_scaler"])

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(out_paths["history_csv"], index=False)

    summary = {
        "model": "LSTM",
        "dataset": dataset_name,
        "profile": profile["name"],
        "features_path": str(features_path),
        "row_count": int(len(df)),
        "feature_count": int(X_train_fit_df.shape[1]),
        "sequence_length": int(profile["sequence_length"]),
        "train_rows": int(len(train)),
        "train_fit_rows": int(len(train_fit)),
        "valid_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "train_seq_rows": int(len(X_train_seq)),
        "valid_seq_rows": int(len(X_valid_seq)),
        "test_seq_rows": int(len(X_test_seq)),
        "train_start": str(train["datetime"].min()),
        "train_end": str(train["datetime"].max()),
        "valid_start": str(valid["datetime"].min()),
        "valid_end": str(valid["datetime"].max()),
        "test_start": str(test_results_df["datetime"].min()),
        "test_end": str(test_results_df["datetime"].max()),
        "epochs_trained": int(len(history.history["loss"])),
        "best_val_loss": float(np.min(history.history["val_loss"])),
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
        "model": "LSTM",
        "data": dataset_name,
        **metrics,
        "test_start": str(test_results_df["datetime"].min()),
        "test_end": str(test_results_df["datetime"].max()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        required=True,
        help="Pilnas arba santykinis kelias iki features CSV"
    )
    args = parser.parse_args()

    train_lstm(args.features)