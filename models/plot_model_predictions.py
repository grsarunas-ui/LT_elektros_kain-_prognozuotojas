from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRED_DIR = Path("data/processed")
PLOT_DIR = PRED_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "15min_clean": {
        "XGBoost": PRED_DIR / "xgb_predictions_15min_clean.csv",
        "MLP": PRED_DIR / "mlp_predictions_15min_clean.csv",
    },
    "15min_extended": {
        "XGBoost": PRED_DIR / "xgb_predictions_15min_extended.csv",
        "MLP": PRED_DIR / "mlp_predictions_15min_extended.csv",
    },
    "hourly_clean": {
        "XGBoost": PRED_DIR / "xgb_predictions_hourly_clean.csv",
        "MLP": PRED_DIR / "mlp_predictions_hourly_clean.csv",
    },
    "hourly_extended": {
        "XGBoost": PRED_DIR / "xgb_predictions_hourly_extended.csv",
        "MLP": PRED_DIR / "mlp_predictions_hourly_extended.csv",
    },
}

MAX_POINTS = 800
TOP_ERROR_COUNT = 30


def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)
    ) * 100


def get_rolling_window(dataset_name: str) -> int:
    if "15min" in dataset_name:
        return 96
    return 24


def load_prediction_file(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"Nerastas failas: {path}")
        return None

    df = pd.read_csv(path)
    required = {"datetime", "price", "predicted_price"}
    if not required.issubset(df.columns):
        print(f"Blogas failo formatas: {path}")
        return None

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["predicted_price"] = pd.to_numeric(df["predicted_price"], errors="coerce")

    df = (
        df.dropna(subset=["datetime", "price", "predicted_price"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    df["error"] = df["predicted_price"] - df["price"]
    df["abs_error"] = df["error"].abs()
    df["squared_error"] = df["error"] ** 2
    df["ape_pct"] = df["abs_error"] / (df["price"].abs() + 1e-6) * 100

    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["date"] = df["datetime"].dt.date
    df["month"] = df["datetime"].dt.to_period("M").astype(str)
    df["week"] = df["datetime"].dt.to_period("W").astype(str)

    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    mae = df["abs_error"].mean()
    rmse = np.sqrt(df["squared_error"].mean())

    ss_res = ((df["price"] - df["predicted_price"]) ** 2).sum()
    ss_tot = ((df["price"] - df["price"].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    corr = df["price"].corr(df["predicted_price"])

    return {
        "rows": len(df),
        "mae": float(mae),
        "median_ae": float(df["abs_error"].median()),
        "rmse": float(rmse),
        "r2": float(r2),
        "corr": float(corr) if pd.notna(corr) else np.nan,
        "smape": float(smape(df["price"], df["predicted_price"])),
        "mape_pct": float(df["ape_pct"].mean()),
        "error_mean": float(df["error"].mean()),
        "error_std": float(df["error"].std()),
        "error_min": float(df["error"].min()),
        "error_max": float(df["error"].max()),
        "abs_error_max": float(df["abs_error"].max()),
        "abs_error_p90": float(df["abs_error"].quantile(0.90)),
        "abs_error_p95": float(df["abs_error"].quantile(0.95)),
        "abs_error_p99": float(df["abs_error"].quantile(0.99)),
        "ape_p90": float(df["ape_pct"].quantile(0.90)),
        "ape_p95": float(df["ape_pct"].quantile(0.95)),
        "ape_p99": float(df["ape_pct"].quantile(0.99)),
        "price_std": float(df["price"].std()),
        "pred_std": float(df["predicted_price"].std()),
        "overprediction_rate": float((df["error"] > 0).mean()),
        "underprediction_rate": float((df["error"] < 0).mean()),
    }


def save_metrics_csv(all_metrics: list[dict]):
    if not all_metrics:
        return

    out_path = PRED_DIR / "model_detailed_metrics.csv"
    pd.DataFrame(all_metrics).sort_values(["dataset", "mae"]).to_csv(out_path, index=False)
    print(f"Išsaugota: {out_path}")


def save_top_errors_csv(dataset_name: str, model_name: str, df: pd.DataFrame):
    out = df.sort_values("abs_error", ascending=False).head(TOP_ERROR_COUNT).copy()
    out_path = PLOT_DIR / f"top_errors_{dataset_name}_{model_name.lower()}.csv"
    out.to_csv(out_path, index=False)
    print(f"Išsaugota: {out_path}")


def save_stat_tables(dataset_name: str, model_name: str, df: pd.DataFrame):
    safe_model = model_name.lower().replace(" ", "_")

    # daily stats
    daily = df.groupby("date", as_index=False).agg(
        mae=("abs_error", "mean"),
        rmse=("squared_error", lambda x: np.sqrt(np.mean(x))),
        bias=("error", "mean"),
        max_abs_error=("abs_error", "max"),
        actual_mean=("price", "mean"),
        pred_mean=("predicted_price", "mean"),
    )
    daily.to_csv(PLOT_DIR / f"{safe_model}_{dataset_name}_daily_stats.csv", index=False)

    # hourly stats
    hourly = df.groupby("hour", as_index=False).agg(
        mae=("abs_error", "mean"),
        rmse=("squared_error", lambda x: np.sqrt(np.mean(x))),
        bias=("error", "mean"),
        error_std=("error", "std"),
        rows=("error", "size"),
    )
    hourly.to_csv(PLOT_DIR / f"{safe_model}_{dataset_name}_hourly_stats.csv", index=False)

    # weekday stats
    weekday = df.groupby("weekday", as_index=False).agg(
        mae=("abs_error", "mean"),
        rmse=("squared_error", lambda x: np.sqrt(np.mean(x))),
        bias=("error", "mean"),
        error_std=("error", "std"),
        rows=("error", "size"),
    )
    weekday.to_csv(PLOT_DIR / f"{safe_model}_{dataset_name}_weekday_stats.csv", index=False)

    # monthly stats
    monthly = df.groupby("month", as_index=False).agg(
        mae=("abs_error", "mean"),
        rmse=("squared_error", lambda x: np.sqrt(np.mean(x))),
        bias=("error", "mean"),
        max_abs_error=("abs_error", "max"),
        rows=("error", "size"),
    )
    monthly.to_csv(PLOT_DIR / f"{safe_model}_{dataset_name}_monthly_stats.csv", index=False)

    # weekly stats
    weekly = df.groupby("week", as_index=False).agg(
        mae=("abs_error", "mean"),
        rmse=("squared_error", lambda x: np.sqrt(np.mean(x))),
        bias=("error", "mean"),
        max_abs_error=("abs_error", "max"),
        rows=("error", "size"),
    )
    weekly.to_csv(PLOT_DIR / f"{safe_model}_{dataset_name}_weekly_stats.csv", index=False)

    # quantiles
    quantiles = pd.DataFrame({
        "metric": ["abs_error", "ape_pct", "error"],
        "p50": [
            df["abs_error"].quantile(0.50),
            df["ape_pct"].quantile(0.50),
            df["error"].quantile(0.50),
        ],
        "p90": [
            df["abs_error"].quantile(0.90),
            df["ape_pct"].quantile(0.90),
            df["error"].quantile(0.90),
        ],
        "p95": [
            df["abs_error"].quantile(0.95),
            df["ape_pct"].quantile(0.95),
            df["error"].quantile(0.95),
        ],
        "p99": [
            df["abs_error"].quantile(0.99),
            df["ape_pct"].quantile(0.99),
            df["error"].quantile(0.99),
        ],
    })
    quantiles.to_csv(PLOT_DIR / f"{safe_model}_{dataset_name}_quantiles.csv", index=False)

    # descriptive stats
    desc = df[["price", "predicted_price", "error", "abs_error", "ape_pct"]].describe().T
    desc.to_csv(PLOT_DIR / f"{safe_model}_{dataset_name}_describe.csv")


def plot_combined(dataset_name: str, model_files: dict[str, Path]):
    loaded = {}

    for model_name, path in model_files.items():
        df = load_prediction_file(path)
        if df is not None:
            loaded[model_name] = df

    if not loaded:
        print(f"Nėra ką braižyti datasetui: {dataset_name}")
        return

    first_model = list(loaded.keys())[0]
    base_df = loaded[first_model][["datetime", "price"]].copy()
    plot_df = base_df.copy()

    for model_name, df_model in loaded.items():
        plot_df = plot_df.merge(
            df_model[["datetime", "predicted_price"]].rename(
                columns={"predicted_price": f"pred_{model_name}"}
            ),
            on="datetime",
            how="left",
        )

    plot_df = plot_df.iloc[:MAX_POINTS].copy()

    plt.figure(figsize=(16, 7))
    plt.plot(plot_df["datetime"], plot_df["price"], label="Faktinė kaina", linewidth=2)

    for col in plot_df.columns:
        if col.startswith("pred_"):
            plt.plot(plot_df["datetime"], plot_df[col], label=col.replace("pred_", ""))

    plt.title(f"Faktinė vs prognozuota kaina ({dataset_name})")
    plt.xlabel("Data")
    plt.ylabel("Kaina")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    out_path = PLOT_DIR / f"combined_{dataset_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Išsaugota: {out_path}")


def plot_individual(dataset_name: str, model_name: str, df: pd.DataFrame):
    safe_model = model_name.lower().replace(" ", "_")
    window = get_rolling_window(dataset_name)
    df_plot = df.iloc[:MAX_POINTS].copy()

    # 1
    plt.figure(figsize=(16, 6))
    plt.plot(df_plot["datetime"], df_plot["price"], label="Faktinė kaina", linewidth=2)
    plt.plot(df_plot["datetime"], df_plot["predicted_price"], label=f"{model_name} prognozė")
    plt.title(f"{model_name}: faktinė vs prognozuota kaina ({dataset_name})")
    plt.xlabel("Data")
    plt.ylabel("Kaina")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{safe_model}_{dataset_name}_actual_vs_pred.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2
    plt.figure(figsize=(16, 5))
    plt.plot(df_plot["datetime"], df_plot["error"])
    plt.axhline(0, linestyle="--")
    plt.title(f"{model_name}: rezidualai laike ({dataset_name})")
    plt.xlabel("Data")
    plt.ylabel("Rezidualas")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{safe_model}_{dataset_name}_residuals_over_time.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3
    plt.figure(figsize=(16, 5))
    plt.plot(df_plot["datetime"], df_plot["abs_error"])
    plt.title(f"{model_name}: absoliuti klaida laike ({dataset_name})")
    plt.xlabel("Data")
    plt.ylabel("Abs error")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{safe_model}_{dataset_name}_abs_error_over_time.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4
    plt.figure(figsize=(10, 6))
    plt.hist(df["error"], bins=60)
    plt.title(f"{model_name}: rezidualų histograma ({dataset_name})")
    plt.xlabel("Rezidualas")
    plt.ylabel("Dažnis")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{safe_model}_{dataset_name}_residual_hist.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5
    plt.figure(figsize=(10, 6))
    plt.hist(df["abs_error"], bins=60)
    plt.title(f"{model_name}: absoliučių klaidų histograma ({dataset_name})")
    plt.xlabel("Abs error")
    plt.ylabel("Dažnis")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{safe_model}_{dataset_name}_abs_error_hist.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 6
    plt.figure(figsize=(10, 6))
    plt.hist(df["ape_pct"], bins=60)
    plt.title(f"{model_name}: APE % histograma ({dataset_name})")
    plt.xlabel("APE %")
    plt.ylabel("Dažnis")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{safe_model}_{dataset_name}_ape_hist.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 7
    plt.figure(figsize=(7, 7))
    plt.scatter(df["price"], df["predicted_price"], s=10, alpha=0.6)
    min_val = min(df["price"].min(), df["predicted_price"].min())
    max_val = max(df["price"].max(), df["predicted_price"].max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.title(f"{model_name}: faktas vs prognozė ({dataset_name})")
    plt.xlabel("Faktinė kaina")
    plt.ylabel("Prognozuota kaina")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{safe_model}_{dataset_name}_scatter_actual_vs_pred.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 8
    plt.figure(figsize=(7, 7))
    plt.scatter(df["price"], df["error"], s=10, alpha=0.6)
    plt.axhline(0, linestyle="--")
    plt.title(f"{model_name}: klaida vs faktinė kaina ({dataset_name})")
    plt.xlabel("Faktinė kaina")
    plt.ylabel("Klaida")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{safe_model}_{dataset_name}_error_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 9
    rolling_mae = df["abs_error"].rolling(window).mean()
    plt.figure(figsize=(16, 5))
    plt.plot(df["datetime"], rolling_mae)
    plt.title(f"{model_name}: rolling MAE ({dataset_name}, window={window})")
    plt.xlabel("Data")
    plt.ylabel("Rolling MAE")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{safe_model}_{dataset_name}_rolling_mae.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 10
    rolling_std = df["error"].rolling(window).std()
    plt.figure(figsize=(16, 5))
    plt.plot(df["datetime"], rolling_std)
    plt.title(f"{model_name}: rolling rezidualų std ({dataset_name}, window={window})")
    plt.xlabel("Data")
    plt.ylabel("Rolling std(error)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{safe_model}_{dataset_name}_rolling_error_std.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 11
    plt.figure(figsize=(16, 5))
    plt.plot(df["datetime"], df["price"].rolling(window).std(), label="Fact std")
    plt.plot(df["datetime"], df["predicted_price"].rolling(window).std(), label="Pred std")
    plt.title(f"{model_name}: rolling std faktas vs prognozė ({dataset_name})")
    plt.xlabel("Data")
    plt.ylabel("Rolling std")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{safe_model}_{dataset_name}_rolling_price_std_compare.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 12
    cumsum_abs = df["abs_error"].cumsum()
    plt.figure(figsize=(16, 5))
    plt.plot(df["datetime"], cumsum_abs)
    plt.title(f"{model_name}: cumulative absolute error ({dataset_name})")
    plt.xlabel("Data")
    plt.ylabel("Cumulative abs error")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{safe_model}_{dataset_name}_cumulative_abs_error.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 13
    by_hour = df.groupby("hour", as_index=False)["abs_error"].mean()
    plt.figure(figsize=(10, 5))
    plt.bar(by_hour["hour"], by_hour["abs_error"])
    plt.title(f"{model_name}: MAE pagal valandą ({dataset_name})")
    plt.xlabel("Valanda")
    plt.ylabel("Mean abs error")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{safe_model}_{dataset_name}_mae_by_hour.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 14
    by_hour_std = df.groupby("hour", as_index=False)["error"].std()
    plt.figure(figsize=(10, 5))
    plt.bar(by_hour_std["hour"], by_hour_std["error"])
    plt.title(f"{model_name}: error std pagal valandą ({dataset_name})")
    plt.xlabel("Valanda")
    plt.ylabel("Std(error)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{safe_model}_{dataset_name}_error_std_by_hour.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 15
    weekday_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    by_weekday = df.groupby("weekday", as_index=False)["abs_error"].mean()
    by_weekday["label"] = by_weekday["weekday"].map(weekday_map)
    plt.figure(figsize=(8, 5))
    plt.bar(by_weekday["label"], by_weekday["abs_error"])
    plt.title(f"{model_name}: MAE pagal savaitės dieną ({dataset_name})")
    plt.xlabel("Savaitės diena")
    plt.ylabel("Mean abs error")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{safe_model}_{dataset_name}_mae_by_weekday.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 16
    daily_mae = df.groupby("date", as_index=False)["abs_error"].mean()
    plt.figure(figsize=(14, 5))
    plt.plot(pd.to_datetime(daily_mae["date"]), daily_mae["abs_error"])
    plt.title(f"{model_name}: dienos MAE ({dataset_name})")
    plt.xlabel("Data")
    plt.ylabel("Daily MAE")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{safe_model}_{dataset_name}_daily_mae.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 17
    daily_max = df.groupby("date", as_index=False)["abs_error"].max()
    plt.figure(figsize=(14, 5))
    plt.plot(pd.to_datetime(daily_max["date"]), daily_max["abs_error"])
    plt.title(f"{model_name}: dienos max abs error ({dataset_name})")
    plt.xlabel("Data")
    plt.ylabel("Max abs error")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{safe_model}_{dataset_name}_daily_max_abs_error.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 18
    top_err = df.sort_values("abs_error", ascending=False).head(TOP_ERROR_COUNT)
    plt.figure(figsize=(16, 6))
    plt.plot(df_plot["datetime"], df_plot["price"], label="Faktinė kaina", linewidth=2)
    plt.plot(df_plot["datetime"], df_plot["predicted_price"], label=f"{model_name} prognozė")
    top_err_plot = top_err[top_err["datetime"].isin(df_plot["datetime"])]
    if not top_err_plot.empty:
        plt.scatter(top_err_plot["datetime"], top_err_plot["price"], s=35, label="Top klaidos")
    plt.title(f"{model_name}: top klaidos pažymėtos grafike ({dataset_name})")
    plt.xlabel("Data")
    plt.ylabel("Kaina")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{safe_model}_{dataset_name}_top_errors_marked.png", dpi=150, bbox_inches="tight")
    plt.close()

    save_top_errors_csv(dataset_name, model_name, df)
    save_stat_tables(dataset_name, model_name, df)


def main():
    all_metrics = []

    for dataset_name, model_files in FILES.items():
        print("\n" + "=" * 70)
        print(f"Braižomas datasetas: {dataset_name}")

        plot_combined(dataset_name, model_files)

        for model_name, path in model_files.items():
            df = load_prediction_file(path)
            if df is None:
                continue

            metrics = compute_metrics(df)
            metrics["dataset"] = dataset_name
            metrics["model"] = model_name
            all_metrics.append(metrics)

            plot_individual(dataset_name, model_name, df)

    save_metrics_csv(all_metrics)

    print("\nBaigta. Visi grafikai ir statistika išsaugoti kataloge:")
    print(PLOT_DIR)


if __name__ == "__main__":
    main()