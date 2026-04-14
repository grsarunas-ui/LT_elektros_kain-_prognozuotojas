import pandas as pd
from pathlib import Path


BASE_PATH = Path(
    "/Users/sarunas/Documents/LT Elektros kainų prognozuotojas/"
    "LT_elektros_kain-_prognozuotojas"
)

INPUT_FILE = BASE_PATH / "data/raw/Commercial transfers/Commercial flows.xlsx"

OUT_HOURLY = BASE_PATH / "data/processed/flows_hourly.csv"
OUT_15MIN = BASE_PATH / "data/processed/flows_15min.csv"


def normalize_header(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pirmą eilutę paverčia stulpelių pavadinimais.
    """
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
    """
    Jei ateityje failo dažnis būtų ne hourly, šita funkcija suvienodins į hourly.
    """
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


def load_flows() -> pd.DataFrame:
    df = pd.read_excel(INPUT_FILE)
    df = normalize_header(df)
    df = clean_column_names(df)

    datetime_col = df.columns[0]

    col_map = {
        "Komercinis srautas Lietuva - Latvija": "flow_lt_lv",
        "Komercinis srautas Lietuva - Švedija": "flow_lt_se",
        "Komercinis srautas Lietuva - Lenkija": "flow_lt_pl",
    }

    missing = [c for c in col_map if c not in df.columns]
    if missing:
        raise ValueError(
            f"Commercial flows faile trūksta stulpelių: {missing}\n"
            f"Rasti stulpeliai: {df.columns.tolist()}"
        )

    keep_cols = [datetime_col] + list(col_map.keys())
    df = df[keep_cols].copy()
    df = df.rename(columns={datetime_col: "datetime", **col_map})

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    flow_cols = ["flow_lt_lv", "flow_lt_se", "flow_lt_pl"]
    for col in flow_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["datetime"])
        .drop_duplicates(subset=["datetime"], keep="last")
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # Paliekam originalius ženklus iš šaltinio.
    # Bendras komercinis srautas.
    df["flow_total"] = (
        df["flow_lt_lv"].fillna(0)
        + df["flow_lt_se"].fillna(0)
        + df["flow_lt_pl"].fillna(0)
    )

    # Papildomas stiprumo signalas be ženklo
    df["flow_abs_total"] = (
        df["flow_lt_lv"].abs().fillna(0)
        + df["flow_lt_se"].abs().fillna(0)
        + df["flow_lt_pl"].abs().fillna(0)
    )

    return df


def build_hourly_dataset() -> pd.DataFrame:
    df = load_flows()
    df_hourly = to_hourly(df)
    return df_hourly


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