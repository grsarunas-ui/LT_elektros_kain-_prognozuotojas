from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


BASE_PATH = Path(
    "/Users/sarunas/Documents/LT Elektros kainų prognozuotojas/"
    "LT_elektros_kain-_prognozuotojas"
)

DATA_DIR = BASE_PATH / "data/processed"
PLOT_DIR = DATA_DIR / "final_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "15min_clean": {
        "XGBoost": DATA_DIR / "xgb_predictions_15min_clean.csv",
        "MLP": DATA_DIR / "mlp_predictions_15min_clean.csv",
    },
    "15min_extended": {
        "XGBoost": DATA_DIR / "xgb_predictions_15min_extended.csv",
        "MLP": DATA_DIR / "mlp_predictions_15min_extended.csv",
    },
    "hourly_clean": {
        "XGBoost": DATA_DIR / "xgb_predictions_hourly_clean.csv",
        "MLP": DATA_DIR / "mlp_predictions_hourly_clean.csv",
    },
    "hourly_extended": {
        "XGBoost": DATA_DIR / "xgb_predictions_hourly_extended.csv",
        "MLP": DATA_DIR / "mlp_predictions_hourly_extended.csv",
    },
}


def load_df(path: Path):
    if not path.exists():
        print(f"Nerastas: {path}")
        return None

    df = pd.read_csv(path)
    required = {"datetime", "price", "predicted_price"}
    if not required.issubset(df.columns):
        print(f"Blogas failo formatas: {path}")
        return None

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return df


def plot_model(df, dataset, model):
    plt.figure(figsize=(14, 6))

    plt.plot(df["datetime"], df["price"], label="Actual", linewidth=2)
    plt.plot(df["datetime"], df["predicted_price"], label="Predicted")

    plt.title(f"{dataset} - {model}")
    plt.xlabel("Date")
    plt.ylabel("Price (EUR/MWh)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    out = PLOT_DIR / f"{dataset}_{model}_vs_actual.png"
    plt.savefig(out, dpi=150)
    plt.close()

    print(f"Saved: {out}")


def plot_summary():
    comp_path = DATA_DIR / "model_comparison.csv"

    if not comp_path.exists():
        print("Nerastas comparison failas")
        return

    df = pd.read_csv(comp_path)

    plt.figure(figsize=(11, 6))

    for _, row in df.iterrows():
        label = f"{row['model']} ({row['data']})"
        plt.bar(label, row["mae"])

    plt.title("Model Comparison (MAE)")
    plt.ylabel("MAE (EUR/MWh)")
    plt.xticks(rotation=35)
    plt.tight_layout()

    out = PLOT_DIR / "model_comparison_bar.png"
    plt.savefig(out, dpi=150)
    plt.close()

    print(f"Saved: {out}")


def main():
    print("\nGenerating model plots...")

    for dataset, models in FILES.items():
        for model_name, path in models.items():
            df = load_df(path)

            if df is None or df.empty:
                print(f"Nėra duomenų: {dataset} {model_name}")
                continue

            plot_model(df, dataset, model_name)

    print("\nGenerating summary chart...")
    plot_summary()

    print("\nDONE. Visi grafikai:")
    print(PLOT_DIR)


if __name__ == "__main__":
    main()