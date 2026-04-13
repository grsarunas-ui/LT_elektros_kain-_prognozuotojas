from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests


BASE_PATH = Path(
    "/Users/sarunas/Documents/LT Elektros kainų prognozuotojas/"
    "LT_elektros_kain-_prognozuotojas"
)

OUT_DIR = BASE_PATH / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ALL_POINTS = OUT_DIR / "weather_lithuania_points_hourly.csv"
OUT_LT_AVG = OUT_DIR / "weather_lithuania_avg_hourly.csv"

# Lietuvos taškai, kad vidurkis nebūtų tik pagal Vilnių
# Pavadinimas, latitude, longitude
LT_POINTS: list[tuple[str, float, float]] = [
    ("Vilnius", 54.6872, 25.2797),
    ("Kaunas", 54.8985, 23.9036),
    ("Klaipeda", 55.7033, 21.1443),
    ("Siauliai", 55.9349, 23.3137),
    ("Panevezys", 55.7348, 24.3575),
    ("Utena", 55.4976, 25.5992),
    ("Alytus", 54.3964, 24.0414),
]

START_DATE = "2022-01-01"
END_DATE = "2026-03-31"

# Official variable names in Open-Meteo docs
HOURLY_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "cloud_cover",
    "shortwave_radiation",
]

API_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_point_weather(name: str, lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Europe/Vilnius",
    }

    response = requests.get(API_URL, params=params, timeout=60)
    response.raise_for_status()
    data: dict[str, Any] = response.json()

    if "hourly" not in data:
        raise ValueError(f"Nėra 'hourly' duomenų atsakyme taškui {name}: {data}")

    hourly = data["hourly"]
    df = pd.DataFrame(hourly)

    if "time" not in df.columns:
        raise ValueError(f"Nėra 'time' stulpelio Open-Meteo atsakyme taškui {name}")

    df["datetime"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["datetime"]).drop(columns=["time"])

    df["location_name"] = name
    df["latitude"] = lat
    df["longitude"] = lon

    ordered_cols = ["datetime", "location_name", "latitude", "longitude"] + HOURLY_VARS
    df = df[ordered_cols].copy()

    return df


def build_all_points_dataset(start_date: str, end_date: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for name, lat, lon in LT_POINTS:
        print(f"Downloading weather: {name} ({lat}, {lon})")
        df = fetch_point_weather(name, lat, lon, start_date, end_date)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["datetime", "location_name"]).reset_index(drop=True)
    return out


def build_lithuania_average(df_all_points: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = HOURLY_VARS

    df_avg = (
        df_all_points.groupby("datetime", as_index=False)[numeric_cols]
        .mean()
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # Papildomas saulės proxy
    if {"shortwave_radiation", "cloud_cover"}.issubset(df_avg.columns):
        df_avg["solar_proxy"] = df_avg["shortwave_radiation"] * (1 - df_avg["cloud_cover"] / 100.0)

    return df_avg


def main():
    print("=== OPEN-METEO WEATHER DOWNLOAD ===")
    print(f"Period: {START_DATE} -> {END_DATE}")
    print(f"Points: {len(LT_POINTS)}")

    df_all = build_all_points_dataset(START_DATE, END_DATE)
    df_avg = build_lithuania_average(df_all)

    df_all.to_csv(OUT_ALL_POINTS, index=False)
    df_avg.to_csv(OUT_LT_AVG, index=False)

    print("\nSaved all points:", OUT_ALL_POINTS)
    print("Shape:", df_all.shape)
    print("Range:", df_all["datetime"].min(), "->", df_all["datetime"].max())
    print(df_all.head())

    print("\nSaved Lithuania average:", OUT_LT_AVG)
    print("Shape:", df_avg.shape)
    print("Range:", df_avg["datetime"].min(), "->", df_avg["datetime"].max())
    print(df_avg.head())


if __name__ == "__main__":
    main()