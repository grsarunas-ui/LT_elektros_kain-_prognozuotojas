import pandas as pd
from pathlib import Path


# =========================
# NUSTATYMAI
# =========================
INPUT_PATH = Path(
    "/Users/sarunas/Documents/LT Elektros kainų prognozuotojas/"
    "LT_elektros_kain-_prognozuotojas/data/raw/"
    "Electricity price Litgrid/Combined electricity prices Litgrid.xlsx"
)

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# nuo kada realiai prasideda 15 min intervalai
CUTOFF_15MIN = "2025-10-01"


# =========================
# 1. UŽKROVIMAS
# =========================
df = pd.read_excel(INPUT_PATH)

print("Pradinė forma:", df.shape)
print("Pradiniai stulpeliai:", df.columns.tolist())

if df.shape[1] < 2:
    raise ValueError("Faile nerasti bent 2 stulpeliai (datetime ir price).")

# pervadinam pirmus 2 stulpelius
df = df.rename(columns={
    df.columns[0]: "datetime",
    df.columns[1]: "price"
})

# =========================
# 2. TIPŲ TVARKYMAS
# =========================
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# pašalinam blogas eilutes
df = df.dropna(subset=["datetime", "price"]).copy()

# rikiuojam
df = df.sort_values("datetime").reset_index(drop=True)

# pašalinam dublikatus
df = df.drop_duplicates(subset="datetime", keep="last").reset_index(drop=True)

print("\nPo bazinio valymo:")
print("Forma:", df.shape)
print("Laikotarpis:", df["datetime"].min(), "->", df["datetime"].max())

if df.empty:
    raise ValueError("Po bazinio valymo neliko duomenų.")

# =========================
# 3. PILNAS RAW DATASETAS (visos datos)
# =========================
df_raw = df.copy()

# intervalų patikra visam rinkiniui
df_raw["time_diff_min"] = df_raw["datetime"].diff().dt.total_seconds() / 60

print("\nDažniausi intervalai minutėmis VISAM rinkiniui:")
print(df_raw["time_diff_min"].value_counts(dropna=True).head(10))

raw_out = OUTPUT_DIR / "master_raw.csv"
df_raw.drop(columns=["time_diff_min"]).to_csv(raw_out, index=False)
print(f"\nIšsaugota: {raw_out}")

# =========================
# 4. 15 MIN DATASETAS (tik nuo 2025-10-01)
# =========================
df_15min = df[df["datetime"] >= CUTOFF_15MIN].copy()
df_15min = df_15min.sort_values("datetime").reset_index(drop=True)

print("\n15 min dataset po cutoff:")
print("Forma:", df_15min.shape)
print("Laikotarpis:", df_15min["datetime"].min(), "->", df_15min["datetime"].max())

if df_15min.empty:
    raise ValueError("Po filtravimo 15 min periodui neliko duomenų.")

# 15 min intervalų patikra
df_15min["time_diff_min"] = df_15min["datetime"].diff().dt.total_seconds() / 60

print("\nDažniausi intervalai minutėmis 15 min rinkiniui:")
print(df_15min["time_diff_min"].value_counts(dropna=True).head(10))

full_15m_range = pd.date_range(
    start=df_15min["datetime"].min(),
    end=df_15min["datetime"].max(),
    freq="15min"
)

missing_15m = full_15m_range.difference(df_15min["datetime"])

print("\nTrūkstami 15 min intervalai:", len(missing_15m))
if len(missing_15m) > 0:
    print("Pirmos trūkstamos datos:")
    print(missing_15m[:10])

min15_out = OUTPUT_DIR / "master_15min.csv"
df_15min.drop(columns=["time_diff_min"]).to_csv(min15_out, index=False)
print(f"Išsaugota: {min15_out}")

# =========================
# 5. HOURLY DATASETAS IŠ VISŲ DATŲ
# =========================
df_hourly = df.copy()
df_hourly["datetime_hour"] = df_hourly["datetime"].dt.floor("h")

df_hourly = (
    df_hourly.groupby("datetime_hour", as_index=False)["price"]
    .mean()
    .rename(columns={"datetime_hour": "datetime"})
    .sort_values("datetime")
    .reset_index(drop=True)
)

hourly_out = OUTPUT_DIR / "master_hourly.csv"
df_hourly.to_csv(hourly_out, index=False)

print(f"\nIšsaugota: {hourly_out}")
print("Hourly forma:", df_hourly.shape)
print("Hourly laikotarpis:", df_hourly["datetime"].min(), "->", df_hourly["datetime"].max())

# =========================
# 6. HOURLY PATIKRA
# =========================
full_hourly_range = pd.date_range(
    start=df_hourly["datetime"].min(),
    end=df_hourly["datetime"].max(),
    freq="h"
)

missing_hourly = full_hourly_range.difference(df_hourly["datetime"])

print("\nTrūkstamos valandos hourly rinkinyje:", len(missing_hourly))
if len(missing_hourly) > 0:
    print("Pirmos trūkstamos valandos:")
    print(missing_hourly[:10])

# =========================
# 7. SUVESTINĖ
# =========================
print("\n=== SUVESTINĖ ===")
print("master_raw.csv      -> visos datos, originalus dažnis")
print("master_15min.csv    -> tik nuo 2025-10-01, 15 min periodas")
print("master_hourly.csv   -> visos datos, agreguota į valandas")