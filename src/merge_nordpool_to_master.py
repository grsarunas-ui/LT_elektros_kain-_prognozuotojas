import os
from glob import glob
from pathlib import Path

import pandas as pd


# =========================
# NUSTATYMAI
# =========================
NORDPOOL_DIR = Path("data/raw/nordpool_raw")

MASTER_15MIN_PATH = Path("data/processed/master_15min.csv")
MASTER_HOURLY_PATH = Path("data/processed/master_hourly.csv")

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NP_15MIN_OUT = OUTPUT_DIR / "nordpool_15min.csv"
NP_HOURLY_OUT = OUTPUT_DIR / "nordpool_hourly.csv"

MASTER_WITH_NP_15MIN_OUT = OUTPUT_DIR / "master_with_nordpool_15min.csv"
MASTER_WITH_NP_HOURLY_OUT = OUTPUT_DIR / "master_with_nordpool_hourly.csv"


# =========================
# 1. NORD POOL FAILŲ APDOROJIMAS
# =========================
def load_nordpool_archive(input_dir: Path) -> pd.DataFrame:
    files = glob(str(input_dir / "*.csv"))
    print(f"Rasta Nord Pool failų: {len(files)}")

    if not files:
        raise FileNotFoundError(f"Nerasta CSV failų kataloge: {input_dir}")

    all_dfs = []

    for file in files:
        print(f"\nSkaitomas: {file}")

        try:
            df = pd.read_csv(file, sep=",", quotechar='"', encoding="utf-8-sig")
        except Exception as e:
            print(f"  -> Nepavyko nuskaityti: {e}")
            continue

        required_cols = ["MTU (UTC)", "Area", "Day-ahead Price (EUR/MWh)"]
        if not all(col in df.columns for col in required_cols):
            print("  -> Trūksta reikalingų stulpelių, failas praleidžiamas")
            print("  -> Columns:", df.columns.tolist())
            continue

        # pvz. "01/01/2021 00:00:00 - 01/01/2021 01:00:00"
        df["datetime"] = df["MTU (UTC)"].astype(str).str.split(" - ").str[0]
        df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True, errors="coerce")

        # Pvz. BZN|LT -> LT
        df["area"] = (
            df["Area"]
            .astype(str)
            .str.replace("BZN|", "", regex=False)
            .str.strip()
        )

        df["price"] = pd.to_numeric(df["Day-ahead Price (EUR/MWh)"], errors="coerce")

        # Pasiliekam tik reikalingas zonas
        df = df[df["area"].isin(["LV", "EE", "SE4", "PL"])]

        df = df[["datetime", "area", "price"]].dropna()

        if df.empty:
            print("  -> Po filtravimo neliko eilučių")
            continue

        # long -> wide
        df = df.pivot_table(
            index="datetime",
            columns="area",
            values="price",
            aggfunc="mean"
        ).reset_index()

        df = df.rename(columns={
            "LV": "lv_price",
            "EE": "ee_price",
            "SE4": "se4_price",
            "PL": "pl_price"
        })

        print(f"  -> Eilučių po pivot: {len(df)}")
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("Nepavyko apdoroti nė vieno Nord Pool failo.")

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all["datetime"] = pd.to_datetime(df_all["datetime"], errors="coerce")
    df_all = df_all.dropna(subset=["datetime"])

    price_cols = [c for c in ["lv_price", "ee_price", "se4_price", "pl_price"] if c in df_all.columns]

    # Sujungiame dublikatus iš pasikartojančių failų
    df_all = (
        df_all.groupby("datetime", as_index=False)[price_cols]
        .mean()
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    print("\nNord Pool 15 min forma:", df_all.shape)
    print("Nord Pool 15 min laikotarpis:", df_all["datetime"].min(), "->", df_all["datetime"].max())

    return df_all


# =========================
# 2. NORD POOL 15 MIN -> HOURLY
# =========================
def make_nordpool_hourly(df_np_15min: pd.DataFrame) -> pd.DataFrame:
    df = df_np_15min.copy()
    df["datetime_hour"] = df["datetime"].dt.floor("h")

    price_cols = [c for c in ["lv_price", "ee_price", "se4_price", "pl_price"] if c in df.columns]

    df_hourly = (
        df.groupby("datetime_hour", as_index=False)[price_cols]
        .mean()
        .rename(columns={"datetime_hour": "datetime"})
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    print("\nNord Pool hourly forma:", df_hourly.shape)
    print("Nord Pool hourly laikotarpis:", df_hourly["datetime"].min(), "->", df_hourly["datetime"].max())

    return df_hourly


# =========================
# 3. MERGE SU 15 MIN MASTER
# =========================
def merge_with_master_15min(master_path: Path, np_15min: pd.DataFrame, out_path: Path):
    df_master = pd.read_csv(master_path)
    df_master["datetime"] = pd.to_datetime(df_master["datetime"], errors="coerce")
    df_master = df_master.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    df_np = np_15min.copy()
    df_np["datetime"] = pd.to_datetime(df_np["datetime"], errors="coerce")
    df_np = df_np.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    print("\n=== MERGE 15 MIN ===")
    print("Master 15 min shape:", df_master.shape)
    print("NP 15 min shape:", df_np.shape)

    df = df_master.merge(df_np, on="datetime", how="left")

    np_cols = ["lv_price", "ee_price", "se4_price", "pl_price"]

    print("\nMissing before fill (15 min):")
    print(df[np_cols].isna().sum())

    for col in np_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].ffill()

    # jei pradžioje trūksta, nukerpam pradžią
    df = df.dropna().reset_index(drop=True)

    print("\nShape after fill/dropna (15 min):", df.shape)
    print("Range:", df["datetime"].min(), "->", df["datetime"].max())

    print("\nMissing after fill (15 min):")
    print(df[np_cols].isna().sum())

    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


# =========================
# 4. MERGE SU HOURLY MASTER
# =========================
def merge_with_master_hourly(master_path: Path, np_hourly: pd.DataFrame, out_path: Path):
    df_master = pd.read_csv(master_path)
    df_master["datetime"] = pd.to_datetime(df_master["datetime"], errors="coerce")
    df_master = df_master.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    df_np = np_hourly.copy()
    df_np["datetime"] = pd.to_datetime(df_np["datetime"], errors="coerce")
    df_np = df_np.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    print("\n=== MERGE HOURLY ===")
    print("Master hourly shape:", df_master.shape)
    print("NP hourly shape:", df_np.shape)

    df = df_master.merge(df_np, on="datetime", how="left")

    np_cols = ["lv_price", "ee_price", "se4_price", "pl_price"]

    print("\nMissing before fill (hourly):")
    print(df[np_cols].isna().sum())

    for col in np_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].ffill()

    df = df.dropna().reset_index(drop=True)

    print("\nShape after fill/dropna (hourly):", df.shape)
    print("Range:", df["datetime"].min(), "->", df["datetime"].max())

    print("\nMissing after fill (hourly):")
    print(df[np_cols].isna().sum())

    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


# =========================
# 5. MAIN
# =========================
if __name__ == "__main__":
    # a) surenkam Nord Pool 15 min archyvą
    df_np_15min = load_nordpool_archive(NORDPOOL_DIR)
    df_np_15min.to_csv(NP_15MIN_OUT, index=False)
    print(f"\nSaved: {NP_15MIN_OUT}")

    # b) darom hourly variantą
    df_np_hourly = make_nordpool_hourly(df_np_15min)
    df_np_hourly.to_csv(NP_HOURLY_OUT, index=False)
    print(f"Saved: {NP_HOURLY_OUT}")

    # c) merge su master 15 min
    merge_with_master_15min(
        master_path=MASTER_15MIN_PATH,
        np_15min=df_np_15min,
        out_path=MASTER_WITH_NP_15MIN_OUT
    )

    # d) merge su master hourly
    merge_with_master_hourly(
        master_path=MASTER_HOURLY_PATH,
        np_hourly=df_np_hourly,
        out_path=MASTER_WITH_NP_HOURLY_OUT
    )

    print("\n=== BAIGTA ===")
    print("Sukurti failai:")
    print(f" - {NP_15MIN_OUT}")
    print(f" - {NP_HOURLY_OUT}")
    print(f" - {MASTER_WITH_NP_15MIN_OUT}")
    print(f" - {MASTER_WITH_NP_HOURLY_OUT}")