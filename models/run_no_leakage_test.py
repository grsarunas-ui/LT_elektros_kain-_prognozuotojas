from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd
from keras import layers
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")

FEATURES = {
    "15min_clean": DATA_DIR / "features_15min_clean.csv",
    "hourly_clean": DATA_DIR / "features_hourly_clean.csv",
}

TRAIN_END = "2026-03-01"
TEST_END = "2026-03-15 23:59:59"


def smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)
    ) * 100


def evaluate(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "smape": float(smape(y_true, y_pred)),
    }


def prepare_data(features_path: Path):
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

    train = df[df["datetime"] < TRAIN_END].copy()
    test = df[(df["datetime"] >= TRAIN_END) & (df["datetime"] <= TEST_END)].copy()

    if train.empty:
        raise ValueError(f"Train rinkinys tuščias: {features_path}")

    if test.empty:
        raise ValueError(f"Test rinkinys tuščias: {features_path}")

    X_train = train.drop(columns=["datetime", "price"])
    y_train = train["price"]

    X_test = test.drop(columns=["datetime", "price"])
    y_test = test["price"]

    return df, train, test, X_train, y_train, X_test, y_test


def run_xgb(dataset_name: str, X_train, y_train, X_test, y_test, test_df):
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate(y_test, y_pred)

    pred_path = DATA_DIR / f"xgb_predictions_{dataset_name}_no_leakage.csv"
    model_path = MODELS_DIR / f"xgb_model_{dataset_name}_no_leakage.pkl"

    out = test_df[["datetime", "price"]].copy()
    out["predicted_price"] = y_pred
    out["abs_error"] = np.abs(out["price"] - out["predicted_price"])
    out.to_csv(pred_path, index=False)

    joblib.dump(model, model_path)

    print(f"\n=== XGBoost ({dataset_name}, no leakage) ===")
    print(f"MAE:   {metrics['mae']:.2f}")
    print(f"RMSE:  {metrics['rmse']:.2f}")
    print(f"R2:    {metrics['r2']:.4f}")
    print(f"sMAPE: {metrics['smape']:.2f}%")

    return {
        "model": "XGBoost_no_leakage",
        "data": dataset_name,
        **metrics,
    }


def run_mlp(dataset_name: str, X_train, y_train, X_test, y_test, test_df):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = keras.Sequential(
        [
            layers.Input(shape=(X_train_scaled.shape[1],)),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=64,
        verbose=0,
        callbacks=[early_stopping],
    )

    y_pred = model.predict(X_test_scaled, verbose=0).flatten()
    metrics = evaluate(y_test, y_pred)

    pred_path = DATA_DIR / f"mlp_predictions_{dataset_name}_no_leakage.csv"
    model_path = MODELS_DIR / f"mlp_model_{dataset_name}_no_leakage.keras"
    scaler_path = MODELS_DIR / f"mlp_scaler_{dataset_name}_no_leakage.pkl"

    out = test_df[["datetime", "price"]].copy()
    out["predicted_price"] = y_pred
    out["abs_error"] = np.abs(out["price"] - out["predicted_price"])
    out.to_csv(pred_path, index=False)

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n=== MLP ({dataset_name}, no leakage) ===")
    print(f"MAE:   {metrics['mae']:.2f}")
    print(f"RMSE:  {metrics['rmse']:.2f}")
    print(f"R2:    {metrics['r2']:.4f}")
    print(f"sMAPE: {metrics['smape']:.2f}%")

    return {
        "model": "MLP_no_leakage",
        "data": dataset_name,
        **metrics,
    }


def run_dataset(features_path: Path, dataset_name: str):
    print("\n" + "=" * 80)
    print(f"TESTUOJAM: {dataset_name}")
    print(f"Failas: {features_path}")

    df, train, test, X_train, y_train, X_test, y_test = prepare_data(features_path)

    print("Bendra forma:", df.shape)
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("Train range:", train["datetime"].min(), "->", train["datetime"].max())
    print("Test range:", test["datetime"].min(), "->", test["datetime"].max())
    print("Feature count:", X_train.shape[1])

    results = []
    results.append(run_xgb(dataset_name, X_train, y_train, X_test, y_test, test))
    results.append(run_mlp(dataset_name, X_train, y_train, X_test, y_test, test))

    return results


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for dataset_name, path in FEATURES.items():
        if path.exists():
            all_results.extend(run_dataset(path, dataset_name))
        else:
            print(f"Nerastas failas: {path}")

    if all_results:
        results_df = pd.DataFrame(all_results).sort_values(["data", "mae"]).reset_index(drop=True)
        out_path = DATA_DIR / "model_comparison_no_leakage.csv"

        print("\n" + "=" * 80)
        print("FINAL COMPARISON (NO LEAKAGE)")
        print(results_df)

        results_df.to_csv(out_path, index=False)
        print(f"\nIšsaugota: {out_path}")
    else:
        print("Nepavyko sugeneruoti rezultatų.")


if __name__ == "__main__":
    main()
