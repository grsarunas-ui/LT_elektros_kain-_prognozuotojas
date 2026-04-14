from pathlib import Path
import pandas as pd
from sqlalchemy import text
from app.db import engine


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def run_query(query: str, params: dict | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)


def check_db_file():
    db_path = Path("data/database/electricity_forecast.db")
    print_section("DB FILE CHECK")

    if db_path.exists():
        print(f"✓ DB failas rastas: {db_path.resolve()}")
        print(f"Dydis: {db_path.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print(f"❌ DB failas nerastas: {db_path.resolve()}")


def check_tables():
    print_section("TABLE CHECK")

    query = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name
    """
    df = run_query(query)
    print(df.to_string(index=False))


def check_versions():
    print_section("DATA VERSIONS")

    query = """
        SELECT
            id,
            version_name,
            description,
            created_at
        FROM data_versions
        ORDER BY id
    """
    df = run_query(query)
    if df.empty:
        print("⚠️ data_versions lentelė tuščia")
    else:
        print(df.to_string(index=False))


def check_row_counts():
    print_section("ROW COUNTS")

    queries = {
        "data_versions": "SELECT COUNT(*) AS row_count FROM data_versions",
        "market_data": "SELECT COUNT(*) AS row_count FROM market_data",
        "predictions": "SELECT COUNT(*) AS row_count FROM predictions",
        "model_metrics": "SELECT COUNT(*) AS row_count FROM model_metrics",
    }

    for name, query in queries.items():
        df = run_query(query)
        print(f"{name}: {int(df.iloc[0, 0])}")


def check_market_summary():
    print_section("MARKET DATA SUMMARY")

    query = """
        SELECT
            dv.version_name,
            md.frequency,
            COUNT(*) AS rows_count,
            MIN(md.datetime) AS min_datetime,
            MAX(md.datetime) AS max_datetime,
            AVG(md.price) AS avg_price
        FROM market_data md
        JOIN data_versions dv ON md.data_version_id = dv.id
        GROUP BY dv.version_name, md.frequency
        ORDER BY dv.version_name, md.frequency
    """
    df = run_query(query)
    if df.empty:
        print("⚠️ market_data lentelė tuščia")
    else:
        print(df.to_string(index=False))


def check_predictions_summary():
    print_section("PREDICTIONS SUMMARY")

    query = """
        SELECT
            dv.version_name,
            p.dataset_name,
            p.model_name,
            COUNT(*) AS rows_count,
            MIN(p.datetime) AS min_datetime,
            MAX(p.datetime) AS max_datetime,
            AVG(p.abs_error) AS mae_from_predictions,
            AVG(p.error) AS bias
        FROM predictions p
        JOIN data_versions dv ON p.data_version_id = dv.id
        GROUP BY dv.version_name, p.dataset_name, p.model_name
        ORDER BY dv.version_name, p.dataset_name, p.model_name
    """
    df = run_query(query)
    if df.empty:
        print("⚠️ predictions lentelė tuščia")
    else:
        print(df.to_string(index=False))


def check_metrics():
    print_section("MODEL METRICS")

    query = """
        SELECT
            dv.version_name,
            mm.dataset_name,
            mm.model_name,
            mm.mae,
            mm.rmse,
            mm.r2,
            mm.smape
        FROM model_metrics mm
        JOIN data_versions dv ON mm.data_version_id = dv.id
        ORDER BY dv.version_name, mm.mae ASC
    """
    df = run_query(query)
    if df.empty:
        print("⚠️ model_metrics lentelė tuščia")
    else:
        print(df.to_string(index=False))


def check_duplicates():
    print_section("DUPLICATE CHECK")

    queries = {
        "market_data duplicates": """
            SELECT COUNT(*) AS duplicate_groups
            FROM (
                SELECT data_version_id, frequency, datetime, COUNT(*) AS cnt
                FROM market_data
                GROUP BY data_version_id, frequency, datetime
                HAVING COUNT(*) > 1
            ) t
        """,
        "predictions duplicates": """
            SELECT COUNT(*) AS duplicate_groups
            FROM (
                SELECT data_version_id, dataset_name, model_name, datetime, COUNT(*) AS cnt
                FROM predictions
                GROUP BY data_version_id, dataset_name, model_name, datetime
                HAVING COUNT(*) > 1
            ) t
        """,
        "model_metrics duplicates": """
            SELECT COUNT(*) AS duplicate_groups
            FROM (
                SELECT data_version_id, dataset_name, model_name, COUNT(*) AS cnt
                FROM model_metrics
                GROUP BY data_version_id, dataset_name, model_name
                HAVING COUNT(*) > 1
            ) t
        """,
    }

    for label, query in queries.items():
        df = run_query(query)
        print(f"{label}: {int(df.iloc[0, 0])}")


def check_nulls():
    print_section("IMPORTANT NULL CHECKS")

    query = """
        SELECT
            frequency,
            SUM(CASE WHEN price IS NULL THEN 1 ELSE 0 END) AS null_price,
            SUM(CASE WHEN lv_price IS NULL THEN 1 ELSE 0 END) AS null_lv_price,
            SUM(CASE WHEN ee_price IS NULL THEN 1 ELSE 0 END) AS null_ee_price,
            SUM(CASE WHEN se4_price IS NULL THEN 1 ELSE 0 END) AS null_se4_price,
            SUM(CASE WHEN pl_price IS NULL THEN 1 ELSE 0 END) AS null_pl_price,
            SUM(CASE WHEN consumption_mw IS NULL THEN 1 ELSE 0 END) AS null_consumption,
            SUM(CASE WHEN production_total_mw IS NULL THEN 1 ELSE 0 END) AS null_production,
            SUM(CASE WHEN flow_total IS NULL THEN 1 ELSE 0 END) AS null_flow_total
        FROM market_data
        GROUP BY frequency
        ORDER BY frequency
    """
    df = run_query(query)
    if df.empty:
        print("⚠️ market_data lentelė tuščia")
    else:
        print(df.to_string(index=False))


def check_sample_rows():
    print_section("SAMPLE MARKET DATA ROWS")

    query = """
        SELECT
            datetime,
            frequency,
            price,
            lv_price,
            ee_price,
            se4_price,
            pl_price,
            consumption_mw,
            production_total_mw,
            flow_total
        FROM market_data
        ORDER BY datetime DESC
        LIMIT 10
    """
    df = run_query(query)
    if df.empty:
        print("⚠️ market_data lentelė tuščia")
    else:
        print(df.to_string(index=False))

    print_section("SAMPLE PREDICTION ROWS")

    query = """
        SELECT
            dataset_name,
            model_name,
            datetime,
            actual_price,
            predicted_price,
            abs_error,
            error
        FROM predictions
        ORDER BY datetime DESC
        LIMIT 10
    """
    df = run_query(query)
    if df.empty:
        print("⚠️ predictions lentelė tuščia")
    else:
        print(df.to_string(index=False))


def check_specific_version(version_name: str = "baseline_2026_03_15"):
    print_section(f"SPECIFIC VERSION CHECK: {version_name}")

    query = """
        SELECT
            dv.version_name,
            md.frequency,
            COUNT(*) AS rows_count,
            MIN(md.datetime) AS min_datetime,
            MAX(md.datetime) AS max_datetime
        FROM market_data md
        JOIN data_versions dv ON md.data_version_id = dv.id
        WHERE dv.version_name = :version_name
        GROUP BY dv.version_name, md.frequency
        ORDER BY md.frequency
    """
    df = run_query(query, {"version_name": version_name})
    if df.empty:
        print(f"⚠️ Nerasta versija arba nėra market_data: {version_name}")
    else:
        print(df.to_string(index=False))

    query = """
        SELECT
            p.dataset_name,
            p.model_name,
            COUNT(*) AS rows_count,
            AVG(p.abs_error) AS mae_from_predictions
        FROM predictions p
        JOIN data_versions dv ON p.data_version_id = dv.id
        WHERE dv.version_name = :version_name
        GROUP BY p.dataset_name, p.model_name
        ORDER BY p.dataset_name, p.model_name
    """
    df = run_query(query, {"version_name": version_name})
    if df.empty:
        print(f"⚠️ Nerasta predictions versijai: {version_name}")
    else:
        print("\nPredictions:")
        print(df.to_string(index=False))

    query = """
        SELECT
            mm.dataset_name,
            mm.model_name,
            mm.mae,
            mm.rmse,
            mm.r2,
            mm.smape
        FROM model_metrics mm
        JOIN data_versions dv ON mm.data_version_id = dv.id
        WHERE dv.version_name = :version_name
        ORDER BY mm.mae ASC
    """
    df = run_query(query, {"version_name": version_name})
    if df.empty:
        print(f"⚠️ Nerasta model_metrics versijai: {version_name}")
    else:
        print("\nMetrics:")
        print(df.to_string(index=False))


def main():
    check_db_file()
    check_tables()
    check_versions()
    check_row_counts()
    check_market_summary()
    check_predictions_summary()
    check_metrics()
    check_duplicates()
    check_nulls()
    check_sample_rows()
    check_specific_version("baseline_2026_03_15")

    print("\n" + "=" * 80)
    print("DB CHECK BAIGTAS")
    print("=" * 80)


if __name__ == "__main__":
    main()