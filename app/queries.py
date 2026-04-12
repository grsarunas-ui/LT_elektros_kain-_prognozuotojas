import pandas as pd
from sqlalchemy import text

from app.db import engine


def get_data_versions() -> pd.DataFrame:
    query = """
        SELECT
            id,
            version_name,
            description,
            created_at
        FROM data_versions
        ORDER BY id
    """
    return pd.read_sql(query, engine)


def get_market_data(version_name: str, frequency: str = "hourly") -> pd.DataFrame:
    query = text("""
        SELECT
            md.datetime,
            md.price,
            md.lv_price,
            md.ee_price,
            md.se4_price,
            md.pl_price,
            md.consumption_mw,
            md.production_total_mw,
            md.flow_lt_lv,
            md.flow_lt_se,
            md.flow_lt_pl,
            md.flow_total,
            md.flow_abs_total
        FROM market_data md
        JOIN data_versions dv ON md.data_version_id = dv.id
        WHERE dv.version_name = :version_name
          AND md.frequency = :frequency
        ORDER BY md.datetime
    """)
    return pd.read_sql(query, engine, params={
        "version_name": version_name,
        "frequency": frequency,
    })


def get_predictions(version_name: str, dataset_name: str, model_name: str) -> pd.DataFrame:
    query = text("""
        SELECT
            p.datetime,
            p.actual_price,
            p.predicted_price,
            p.abs_error,
            p.error
        FROM predictions p
        JOIN data_versions dv ON p.data_version_id = dv.id
        WHERE dv.version_name = :version_name
          AND p.dataset_name = :dataset_name
          AND p.model_name = :model_name
        ORDER BY p.datetime
    """)
    return pd.read_sql(query, engine, params={
        "version_name": version_name,
        "dataset_name": dataset_name,
        "model_name": model_name,
    })


def get_model_metrics(version_name: str) -> pd.DataFrame:
    query = text("""
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
    """)
    return pd.read_sql(query, engine, params={"version_name": version_name})


def get_prediction_summary(version_name: str) -> pd.DataFrame:
    query = text("""
        SELECT
            p.dataset_name,
            p.model_name,
            COUNT(*) AS rows_count,
            AVG(p.abs_error) AS mae_from_predictions,
            MAX(p.abs_error) AS max_abs_error,
            AVG(p.error) AS bias
        FROM predictions p
        JOIN data_versions dv ON p.data_version_id = dv.id
        WHERE dv.version_name = :version_name
        GROUP BY p.dataset_name, p.model_name
        ORDER BY mae_from_predictions ASC
    """)
    return pd.read_sql(query, engine, params={"version_name": version_name})