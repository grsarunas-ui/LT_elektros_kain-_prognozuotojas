import pandas as pd
import numpy as np
from pathlib import Path


INPUT_15MIN = Path("data/processed/master_with_nordpool_15min.csv")
INPUT_HOURLY = Path("data/processed/master_with_nordpool_hourly.csv")

OUTPUT_15MIN_CLEAN = Path("data/processed/features_15min_clean.csv")
OUTPUT_15MIN_EXTENDED = Path("data/processed/features_15min_extended.csv")

OUTPUT_HOURLY_CLEAN = Path("data/processed/features_hourly_clean.csv")
OUTPUT_HOURLY_EXTENDED = Path("data/processed/features_hourly_extended.csv")

LITGRID_15MIN = Path("data/processed/litgrid_features_15min.csv")
LITGRID_HOURLY = Path("data/processed/litgrid_features_hourly.csv")

FLOWS_15MIN = Path("data/processed/flows_15min.csv")
FLOWS_HOURLY = Path("data/processed/flows_hourly.csv")


def detect_frequency(df: pd.DataFrame) -> str:
    diffs = df["datetime"].diff().dropna()
    if diffs.empty:
        return "hourly"

    mode_diff = diffs.mode().iloc[0]
    if mode_diff <= pd.Timedelta(minutes=15):
        return "15min"
    return "hourly"


def get_time_config(freq: str) -> dict:
    if freq == "15min":
        return {
            "price_lags": [1, 4, 8, 96, 192, 672],
            "lag_24h": 96,
            "lag_48h": 192,
            "lag_7d": 672,
            "roll_24h": 96,
            "roll_7d": 672,
        }
    else:
        return {
            "price_lags": [1, 2, 3, 24, 48, 72, 168],
            "lag_24h": 24,
            "lag_48h": 48,
            "lag_7d": 168,
            "roll_24h": 24,
            "roll_7d": 168,
        }


def add_time_features(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = df.copy()

    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["weekday"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month
    df["dayofyear"] = df["datetime"].dt.dayofyear
    df["weekofyear"] = df["datetime"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    df["hour_week"] = df["weekday"] * 24 + df["hour"]
    df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    return df


def add_price_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()

    for lag in cfg["price_lags"]:
        df[f"lag_{lag}"] = df["price"].shift(lag)

    if "lag_1" in df.columns:
        df["price_diff_1"] = df["price"].shift(1) - df["price"].shift(2)

    if f"lag_{cfg['lag_24h']}" in df.columns:
        df["price_diff_24h"] = df["lag_1"] - df[f"lag_{cfg['lag_24h']}"]
        df["trend_24h"] = df["lag_1"] - df[f"lag_{cfg['lag_24h']}"]

    if f"lag_{cfg['lag_7d']}" in df.columns:
        df["trend_7d"] = df["lag_1"] - df[f"lag_{cfg['lag_7d']}"]

    df["rolling_mean_24h"] = df["price"].shift(1).rolling(cfg["roll_24h"]).mean()
    df["rolling_mean_7d"] = df["price"].shift(1).rolling(cfg["roll_7d"]).mean()

    df["rolling_std_24h"] = df["price"].shift(1).rolling(cfg["roll_24h"]).std()
    df["rolling_min_24h"] = df["price"].shift(1).rolling(cfg["roll_24h"]).min()
    df["rolling_max_24h"] = df["price"].shift(1).rolling(cfg["roll_24h"]).max()

    df["price_vs_mean24h"] = df["lag_1"] / (df["rolling_mean_24h"] + 1e-6)
    df["price_vs_mean7d"] = df["lag_1"] / (df["rolling_mean_7d"] + 1e-6)

    df["price_spike_flag"] = (df["lag_1"] > df["rolling_mean_24h"] * 1.5).astype(int)

    df["trend_short"] = df["lag_1"] - df["price"].shift(3)
    df["trend_medium"] = df["lag_1"] - df["price"].shift(6)

    df["price_zscore_24h"] = (
        (df["lag_1"] - df["rolling_mean_24h"]) /
        (df["rolling_std_24h"] + 1e-6)
    )

    df["range_position_24h"] = (
        (df["lag_1"] - df["rolling_min_24h"]) /
        (df["rolling_max_24h"] - df["rolling_min_24h"] + 1e-6)
    )

    return df


def add_nordpool_features(df: pd.DataFrame, cfg: dict, include_spreads: bool = True) -> pd.DataFrame:
    df = df.copy()

    nordpool_cols = ["lv_price", "ee_price", "se4_price", "pl_price"]

    for col in nordpool_cols:
        if col not in df.columns:
            continue

        df[col] = pd.to_numeric(df[col], errors="coerce")

        df[f"{col}_lag_24h"] = df[col].shift(cfg["lag_24h"])
        df[f"{col}_lag_48h"] = df[col].shift(cfg["lag_48h"])
        df[f"{col}_lag_7d"] = df[col].shift(cfg["lag_7d"])

        df[f"{col}_rolling_mean_24h"] = df[col].shift(1).rolling(cfg["roll_24h"]).mean()
        df[f"{col}_rolling_std_24h"] = df[col].shift(1).rolling(cfg["roll_24h"]).std()

        df[f"{col}_diff_24h"] = df[col].shift(1) - df[col].shift(cfg["lag_24h"])

    if include_spreads:
        lt_lag_1 = df["price"].shift(1)

        if "lv_price" in df.columns:
            df["spread_lv"] = lt_lag_1 - df["lv_price"].shift(1)
        if "ee_price" in df.columns:
            df["spread_ee"] = lt_lag_1 - df["ee_price"].shift(1)
        if "se4_price" in df.columns:
            df["spread_se4"] = lt_lag_1 - df["se4_price"].shift(1)
        if "pl_price" in df.columns:
            df["spread_pl"] = lt_lag_1 - df["pl_price"].shift(1)

    drop_cols = [c for c in nordpool_cols if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


def add_litgrid_features(df: pd.DataFrame, freq: str, cfg: dict) -> pd.DataFrame:
    df = df.copy()

    litgrid_path = LITGRID_15MIN if freq == "15min" else LITGRID_HOURLY

    if not litgrid_path.exists():
        print(f"⚠️ Litgrid failas nerastas: {litgrid_path}")
        return df

    lit = pd.read_csv(litgrid_path)
    lit["datetime"] = pd.to_datetime(lit["datetime"], errors="coerce")
    lit = lit.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    df = df.merge(lit, on="datetime", how="left")

    if "consumption_mw" in df.columns and "production_total_mw" in df.columns:
        df["net_load_mw"] = df["consumption_mw"] - df["production_total_mw"]

    base_cols = [
        "consumption_mw",
        "production_total_mw",
        "net_load_mw",
    ]

    existing_base_cols = [c for c in base_cols if c in df.columns]

    for col in existing_base_cols:
        df[f"{col}_lag_1"] = df[col].shift(1)
        df[f"{col}_lag_24h"] = df[col].shift(cfg["lag_24h"])
        df[f"{col}_lag_48h"] = df[col].shift(cfg["lag_48h"])
        df[f"{col}_lag_7d"] = df[col].shift(cfg["lag_7d"])
        df[f"{col}_rolling_mean_24h"] = df[col].shift(1).rolling(cfg["roll_24h"]).mean()
        df[f"{col}_rolling_std_24h"] = df[col].shift(1).rolling(cfg["roll_24h"]).std()

    if "consumption_mw" in df.columns:
        df["consumption_diff_24h"] = df["consumption_mw"].shift(1) - df["consumption_mw"].shift(cfg["lag_24h"])

    if "production_total_mw" in df.columns:
        df["production_diff_24h"] = df["production_total_mw"].shift(1) - df["production_total_mw"].shift(cfg["lag_24h"])

    if "net_load_mw" in df.columns:
        df["net_load_diff_24h"] = df["net_load_mw"].shift(1) - df["net_load_mw"].shift(cfg["lag_24h"])

    df = df.drop(columns=existing_base_cols, errors="ignore")

    return df


def add_flows_features(df: pd.DataFrame, freq: str, cfg: dict) -> pd.DataFrame:
    df = df.copy()

    flows_path = FLOWS_15MIN if freq == "15min" else FLOWS_HOURLY

    if not flows_path.exists():
        print(f"⚠️ Flows failas nerastas: {flows_path}")
        return df

    flows = pd.read_csv(flows_path)
    flows["datetime"] = pd.to_datetime(flows["datetime"], errors="coerce")
    flows = flows.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    df = df.merge(flows, on="datetime", how="left")

    base_cols = [
        "flow_lt_lv",
        "flow_lt_se",
        "flow_lt_pl",
        "flow_total",
        "flow_abs_total",
    ]

    existing_base_cols = [c for c in base_cols if c in df.columns]

    for col in existing_base_cols:
        df[f"{col}_lag_1"] = df[col].shift(1)
        df[f"{col}_lag_24h"] = df[col].shift(cfg["lag_24h"])
        df[f"{col}_lag_48h"] = df[col].shift(cfg["lag_48h"])
        df[f"{col}_lag_7d"] = df[col].shift(cfg["lag_7d"])
        df[f"{col}_rolling_mean_24h"] = df[col].shift(1).rolling(cfg["roll_24h"]).mean()
        df[f"{col}_rolling_std_24h"] = df[col].shift(1).rolling(cfg["roll_24h"]).std()

    if "flow_total" in df.columns:
        df["flow_total_diff_24h"] = df["flow_total"].shift(1) - df["flow_total"].shift(cfg["lag_24h"])

    if "flow_abs_total" in df.columns:
        df["flow_abs_total_diff_24h"] = df["flow_abs_total"].shift(1) - df["flow_abs_total"].shift(cfg["lag_24h"])

    df = df.drop(columns=existing_base_cols, errors="ignore")

    return df


def create_features(df: pd.DataFrame, mode: str = "clean") -> pd.DataFrame:
    if mode not in {"clean", "extended"}:
        raise ValueError("mode turi būti 'clean' arba 'extended'")

    df = df.copy()

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    freq = detect_frequency(df)
    cfg = get_time_config(freq)

    print(f"Detected frequency: {freq}")
    print(f"Mode: {mode}")

    df = add_time_features(df, freq)
    df = add_price_features(df, cfg)
    df = add_litgrid_features(df, freq, cfg)
    df = add_flows_features(df, freq, cfg)

    if mode == "extended":
        df = add_nordpool_features(df, cfg, include_spreads=True)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    return df


def process_file(input_path: Path, output_path: Path, mode: str) -> None:
    if not input_path.exists():
        print(f"Nerastas failas: {input_path}")
        return

    df = pd.read_csv(input_path)

    print("\n" + "=" * 80)
    print(f"Apdorojamas: {input_path}")
    print(f"Režimas: {mode}")
    print("Pradinė forma:", df.shape)
    print("Stulpeliai:", df.columns.tolist())

    df_features = create_features(df, mode=mode)

    print("Po feature engineering:", df_features.shape)
    print("Laikotarpis:", df_features["datetime"].min(), "->", df_features["datetime"].max())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False)

    print(f"✓ Išsaugota: {output_path}")


if __name__ == "__main__":
    process_file(INPUT_15MIN, OUTPUT_15MIN_CLEAN, mode="clean")
    process_file(INPUT_15MIN, OUTPUT_15MIN_EXTENDED, mode="extended")
    process_file(INPUT_HOURLY, OUTPUT_HOURLY_CLEAN, mode="clean")
    process_file(INPUT_HOURLY, OUTPUT_HOURLY_EXTENDED, mode="extended")