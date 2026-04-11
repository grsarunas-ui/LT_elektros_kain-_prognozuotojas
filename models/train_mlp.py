import argparse
from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd
from keras import layers
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


TEST_DAYS = 14


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


def train_mlp(features_path: str):
    features_path = Path(features_path)
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

    print("\n==============================")
    print(f"DATA: {features_path}")
    print("Shape:", df.shape)
    print("Range:", df["datetime"].min(), "->", df["datetime"].max())

    train, test, test_start, test_end = get_dynamic_split(df, test_days=TEST_DAYS)

    if train.empty:
        raise ValueError("Train rinkinys tuščias. Per mažai istorinių duomenų.")
    if test.empty:
        raise ValueError("Test rinkinys tuščias. Per mažai naujausių duomenų.")

    print("\nDynamic split:")
    print("Test days:", TEST_DAYS)
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("Train range:", train["datetime"].min(), "->", train["datetime"].max())
    print("Test range:", test["datetime"].min(), "->", test["datetime"].max())

    X_train = train.drop(columns=["datetime", "price"])
    y_train = train["price"]

    X_test = test.drop(columns=["datetime", "price"])
    y_test = test["price"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = keras.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse"
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=64,
        verbose=0,
        callbacks=[early_stopping]
    )

    y_pred = model.predict(X_test_scaled, verbose=0).flatten()

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    smape_val = smape(y_test, y_pred)

    print("\n=== MLP ===")
    print(f"MAE:   {mae:.2f}")
    print(f"RMSE:  {rmse:.2f}")
    print(f"R2:    {r2:.4f}")
    print(f"sMAPE: {smape_val:.2f}%")

    dataset_name = features_path.stem.replace("features_", "")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / f"mlp_predictions_{dataset_name}.csv"
    model_path = Path("models") / f"mlp_model_{dataset_name}.keras"
    scaler_path = Path("models") / f"mlp_scaler_{dataset_name}.pkl"

    out = test[["datetime", "price"]].copy()
    out["predicted_price"] = y_pred
    out["abs_error"] = np.abs(out["price"] - out["predicted_price"])
    out.to_csv(pred_path, index=False)

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Išsaugota: {pred_path}")
    print(f"Išsaugota: {model_path}")
    print(f"Išsaugota: {scaler_path}")

    return {
        "model": "MLP",
        "data": dataset_name,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "smape": float(smape_val),
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

    train_mlp(args.features)