from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# NUSTATYMAI
# =========================================================
BASE_PATH = Path(
    "/Users/sarunas/Documents/LT Elektros kainų prognozuotojas/"
    "LT_elektros_kain-_prognozuotojas"
)

PROCESSED_DIR = BASE_PATH / "data" / "processed"
REPORTS_DIR = BASE_PATH / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FILES_TO_ANALYZE = [
    "master_raw.csv",
    "master_15min.csv",
    "master_hourly.csv",
    "weather_lithuania_points_hourly.csv",
    "weather_lithuania_avg_hourly.csv",
    "litgrid_features_hourly.csv",
    "litgrid_features_15min.csv",
    "flows_hourly.csv",
    "flows_15min.csv",
    "nordpool_15min.csv",
    "nordpool_hourly.csv",
    "master_with_nordpool_15min.csv",
    "master_with_nordpool_hourly.csv",
]

SUMMARY_CSV = REPORTS_DIR / "dataset_summary.csv"
COLUMN_QUALITY_CSV = REPORTS_DIR / "dataset_column_quality.csv"
NUMERIC_STATS_CSV = REPORTS_DIR / "dataset_numeric_stats.csv"
REPORT_TXT = REPORTS_DIR / "dataset_report.txt"


# =========================================================
# PAGALBINĖS FUNKCIJOS
# =========================================================
def try_read_csv(file_path: Path) -> pd.DataFrame:
    """
    Bando nuskaityti CSV saugiai.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise RuntimeError(f"Nepavyko nuskaityti failo {file_path.name}: {e}") from e


def coerce_datetime_if_exists(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    df = df.copy()
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def get_time_diff_summary(df: pd.DataFrame, datetime_col: str = "datetime") -> tuple[str, str]:
    """
    Grąžina:
    - dažniausius intervalus tekstu
    - aptiktą dominuojantį dažnį
    """
    if datetime_col not in df.columns:
        return "Nėra datetime stulpelio", "N/A"

    dt = pd.to_datetime(df[datetime_col], errors="coerce").dropna().sort_values()
    if len(dt) < 2:
        return "Per mažai datetime įrašų", "N/A"

    diffs = dt.diff().dropna().dt.total_seconds() / 60.0
    if diffs.empty:
        return "Nepavyko apskaičiuoti intervalų", "N/A"

    vc = diffs.value_counts().head(10)
    freq_text = "; ".join([f"{idx:.0f} min -> {val} kartų" for idx, val in vc.items()])

    most_common = vc.index[0]
    if np.isclose(most_common, 15):
        detected = "15min"
    elif np.isclose(most_common, 60):
        detected = "hourly"
    else:
        detected = f"{most_common:.0f}min"

    return freq_text, detected


def count_zero_values(series: pd.Series) -> int:
    """
    Skaičiuoja nulines reikšmes tik skaitiniams stulpeliams.
    """
    s = pd.to_numeric(series, errors="coerce")
    return int((s == 0).sum())


def analyze_file(file_path: Path) -> tuple[dict, list[dict], list[dict]]:
    """
    Grąžina:
    - bendrą failo suvestinę
    - stulpelių kokybės įrašus
    - skaitinių stulpelių statistikas
    """
    df = try_read_csv(file_path)
    df = coerce_datetime_if_exists(df, "datetime")

    rows, cols = df.shape
    file_size_kb = round(file_path.stat().st_size / 1024, 2)

    has_datetime = "datetime" in df.columns

    datetime_min = df["datetime"].min() if has_datetime else pd.NaT
    datetime_max = df["datetime"].max() if has_datetime else pd.NaT
    datetime_nulls = int(df["datetime"].isna().sum()) if has_datetime else None
    datetime_duplicates = int(df["datetime"].duplicated().sum()) if has_datetime else None

    time_diff_summary, detected_frequency = get_time_diff_summary(df, "datetime")

    total_missing = int(df.isna().sum().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    total_zero_values = int(sum(count_zero_values(df[col]) for col in numeric_cols))

    summary_row = {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "rows": rows,
        "columns": cols,
        "file_size_kb": file_size_kb,
        "has_datetime": has_datetime,
        "datetime_min": datetime_min,
        "datetime_max": datetime_max,
        "datetime_nulls": datetime_nulls,
        "datetime_duplicates": datetime_duplicates,
        "detected_frequency": detected_frequency,
        "time_diff_summary": time_diff_summary,
        "total_missing_values": total_missing,
        "total_zero_values_numeric": total_zero_values,
        "numeric_columns_count": len(numeric_cols),
        "all_columns": ", ".join(map(str, df.columns.tolist())),
    }

    column_quality_rows: list[dict] = []
    numeric_stats_rows: list[dict] = []

    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_pct = round((missing_count / rows) * 100, 4) if rows else 0.0
        duplicated_count = int(df[col].duplicated().sum()) if col != "datetime" else None

        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        zero_count = count_zero_values(df[col]) if is_numeric else None
        zero_pct = round((zero_count / rows) * 100, 4) if (is_numeric and rows) else None
        unique_count = int(df[col].nunique(dropna=True))

        column_quality_rows.append({
            "file_name": file_path.name,
            "column_name": col,
            "dtype": str(df[col].dtype),
            "rows": rows,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "zero_count": zero_count,
            "zero_pct": zero_pct,
            "unique_count": unique_count,
            "duplicated_count": duplicated_count,
        })

        if is_numeric:
            s = pd.to_numeric(df[col], errors="coerce")
            numeric_stats_rows.append({
                "file_name": file_path.name,
                "column_name": col,
                "count": int(s.count()),
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "q25": s.quantile(0.25),
                "median": s.median(),
                "q75": s.quantile(0.75),
                "max": s.max(),
            })

    return summary_row, column_quality_rows, numeric_stats_rows


def format_value(v) -> str:
    if pd.isna(v):
        return "N/A"
    return str(v)


def write_text_report(
    summary_df: pd.DataFrame,
    column_df: pd.DataFrame,
    numeric_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines: list[str] = []

    lines.append("DUOMENU RINKINIU KOKYBES ATASKAITA")
    lines.append("=" * 80)
    lines.append("")

    for _, row in summary_df.iterrows():
        file_name = row["file_name"]
        lines.append(f"FAILAS: {file_name}")
        lines.append("-" * 80)
        lines.append(f"Eiluciu skaicius: {format_value(row['rows'])}")
        lines.append(f"Stulpeliu skaicius: {format_value(row['columns'])}")
        lines.append(f"Failo dydis (KB): {format_value(row['file_size_kb'])}")
        lines.append(f"Yra datetime stulpelis: {format_value(row['has_datetime'])}")
        lines.append(f"Laikotarpio pradzia: {format_value(row['datetime_min'])}")
        lines.append(f"Laikotarpio pabaiga: {format_value(row['datetime_max'])}")
        lines.append(f"Datetime null reiksmės: {format_value(row['datetime_nulls'])}")
        lines.append(f"Datetime dublikatai: {format_value(row['datetime_duplicates'])}")
        lines.append(f"Aptiktas daznis: {format_value(row['detected_frequency'])}")
        lines.append(f"Dažniausi intervalai: {format_value(row['time_diff_summary'])}")
        lines.append(f"Bendras trukstamu reikšmių skaičius: {format_value(row['total_missing_values'])}")
        lines.append(f"Bendras nuliniu reikšmių skaičius skaitiniuose stulpeliuose: {format_value(row['total_zero_values_numeric'])}")
        lines.append("")

        sub_cols = column_df[column_df["file_name"] == file_name].copy()
        if not sub_cols.empty:
            lines.append("STULPELIU KOKYBE:")
            for _, c in sub_cols.iterrows():
                lines.append(
                    f"  - {c['column_name']} | dtype={c['dtype']} | "
                    f"missing={c['missing_count']} ({c['missing_pct']}%) | "
                    f"zeros={c['zero_count']} | unique={c['unique_count']}"
                )
            lines.append("")

        sub_num = numeric_df[numeric_df["file_name"] == file_name].copy()
        if not sub_num.empty:
            lines.append("SKAITINIU STULPELIU STATISTIKA:")
            for _, n in sub_num.iterrows():
                lines.append(
                    f"  - {n['column_name']} | count={n['count']} | mean={n['mean']:.4f} | "
                    f"std={n['std']:.4f} | min={n['min']:.4f} | median={n['median']:.4f} | max={n['max']:.4f}"
                )
            lines.append("")

        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# MAIN
# =========================================================
def main():
    summary_rows: list[dict] = []
    column_quality_rows: list[dict] = []
    numeric_stats_rows: list[dict] = []

    print("=== PRADEDAMA DUOMENU RINKINIU ANALIZE ===")

    for file_name in FILES_TO_ANALYZE:
        file_path = PROCESSED_DIR / file_name

        if not file_path.exists():
            print(f"[WARN] Nerastas failas: {file_path}")
            continue

        print(f"Analizuojamas: {file_name}")

        try:
            summary_row, col_rows, num_rows = analyze_file(file_path)
            summary_rows.append(summary_row)
            column_quality_rows.extend(col_rows)
            numeric_stats_rows.extend(num_rows)
        except Exception as e:
            print(f"[ERROR] {file_name}: {e}")

    if not summary_rows:
        raise RuntimeError("Nepavyko apdoroti nei vieno failo.")

    summary_df = pd.DataFrame(summary_rows)
    column_df = pd.DataFrame(column_quality_rows)
    numeric_df = pd.DataFrame(numeric_stats_rows)

    summary_df = summary_df.sort_values("file_name").reset_index(drop=True)
    if not column_df.empty:
        column_df = column_df.sort_values(["file_name", "column_name"]).reset_index(drop=True)
    if not numeric_df.empty:
        numeric_df = numeric_df.sort_values(["file_name", "column_name"]).reset_index(drop=True)

    summary_df.to_csv(SUMMARY_CSV, index=False)
    column_df.to_csv(COLUMN_QUALITY_CSV, index=False)
    numeric_df.to_csv(NUMERIC_STATS_CSV, index=False)

    write_text_report(summary_df, column_df, numeric_df, REPORT_TXT)

    print("\n=== BAIGTA ===")
    print(f"Išsaugota suvestinė: {SUMMARY_CSV}")
    print(f"Išsaugota stulpelių kokybė: {COLUMN_QUALITY_CSV}")
    print(f"Išsaugota statistika: {NUMERIC_STATS_CSV}")
    print(f"Išsaugota tekstinė ataskaita: {REPORT_TXT}")

    print("\nTrumpa suvestinė:")
    print(summary_df[
        [
            "file_name",
            "rows",
            "columns",
            "datetime_min",
            "datetime_max",
            "detected_frequency",
            "total_missing_values",
            "total_zero_values_numeric",
        ]
    ])


if __name__ == "__main__":
    main()