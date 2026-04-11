import pandas as pd
from pathlib import Path


BASE_PATH = Path(
    "/Users/sarunas/Documents/LT Elektros kainų prognozuotojas/"
    "LT_elektros_kain-_prognozuotojas"
)

CONSUMPTION_FILE = BASE_PATH / "data/raw/Electricity consumption data Litgrid/Electricity consumption data combined.xlsx"
PRODUCTION_FILE = BASE_PATH / "data/raw/Production data Litgrid/Factual production data.xlsx"

OUT_HOURLY = BASE_PATH / "data/processed/litgrid_features_hourly.csv"
OUT_15MIN = BASE_PATH / "data/processed/litgrid_features_15min.csv"


def normalize_header(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols = df.iloc[0].tolist()
    df = df.iloc[1:].copy()
    df.columns = new_cols
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    value_cols = [c for c in df.columns if c != "datetime"]
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["datetime"] = df["datetime"].dt.floor("h")

    df = (
        df.groupby("datetime", as_index=False)[value_cols]
        .mean()
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    return df


def load_consumption() -> pd.DataFrame:
    df = pd.read_excel(CONSUMPTION_FILE)
    df = normalize_header(df)
    df = clean_column_names(df)

    datetime_col = df.columns[0]
    actual_consumption_col = "Faktinis nacionalinis Elektros energijos vartojimas"

    if actual_consumption_col not in df.columns:
        raise ValueError(
            f"Nerastas stulpelis '{actual_consumption_col}' consumption faile.\n"
            f"Rasti stulpeliai: {df.columns.tolist()}"
        )

    df = df[[datetime_col, actual_consumption_col]].copy()
    df = df.rename(columns={
        datetime_col: "datetime",
        actual_consumption_col: "consumption_mw",
    })

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["consumption_mw"] = pd.to_numeric(df["consumption_mw"], errors="coerce")

    df = (
        df.dropna(subset=["datetime", "consumption_mw"])
        .drop_duplicates(subset=["datetime"], keep="last")
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    return df


def load_production() -> pd.DataFrame:
    df = pd.read_excel(PRODUCTION_FILE)
    df = normalize_header(df)
    df = clean_column_names(df)

    datetime_col = df.columns[0]
    total_production_col = "Faktinė nacionalinė elektros energijos gamyba"

    if total_production_col not in df.columns:
        raise ValueError(
            f"Nerastas stulpelis '{total_production_col}' production faile.\n"
            f"Rasti stulpeliai: {df.columns.tolist()}"
        )

    df = df[[datetime_col, total_production_col]].copy()
    df = df.rename(columns={
        datetime_col: "datetime",
        total_production_col: "production_total_mw",
    })

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["production_total_mw"] = pd.to_numeric(df["production_total_mw"], errors="coerce")

    df = (
        df.dropna(subset=["datetime", "production_total_mw"])
        .drop_duplicates(subset=["datetime"], keep="last")
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    return df


def build_hourly_dataset() -> pd.DataFrame:
    cons = load_consumption()
    prod = load_production()

    cons_hourly = to_hourly(cons)
    prod_hourly = to_hourly(prod)

    df = cons_hourly.merge(prod_hourly, on="datetime", how="inner")
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def build_15min_dataset(df_hourly: pd.DataFrame) -> pd.DataFrame:
    df = df_hourly.copy()
    df = df.set_index("datetime").sort_index()

    df_15 = df.resample("15min").ffill().reset_index()
    return df_15


def main():
    OUT_HOURLY.parent.mkdir(parents=True, exist_ok=True)

    df_hourly = build_hourly_dataset()
    df_15min = build_15min_dataset(df_hourly)

    df_hourly.to_csv(OUT_HOURLY, index=False)
    df_15min.to_csv(OUT_15MIN, index=False)

    print("\nSaved hourly:", OUT_HOURLY)
    print("Shape:", df_hourly.shape)
    print("Range:", df_hourly["datetime"].min(), "->", df_hourly["datetime"].max())
    print(df_hourly.head())

    print("\nSaved 15min:", OUT_15MIN)
    print("Shape:", df_15min.shape)
    print("Range:", df_15min["datetime"].min(), "->", df_15min["datetime"].max())
    print(df_15min.head())


if __name__ == "__main__":
    main()