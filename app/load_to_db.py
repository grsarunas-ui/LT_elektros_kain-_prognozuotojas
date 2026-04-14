from __future__ import annotations
import argparse
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd

from app.db import Base, SessionLocal, engine
from app.models import (
    DataVersion,
    MarketData,
    Prediction,
    ModelMetric,
    FeatureImportance,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


MARKET_FILES = {
    "hourly": {
        "master": PROCESSED_DIR / "master_with_nordpool_hourly.csv",
        "litgrid": PROCESSED_DIR / "litgrid_features_hourly.csv",
        "flows": PROCESSED_DIR / "flows_hourly.csv",
    },
    "15min": {
        "master": PROCESSED_DIR / "master_with_nordpool_15min.csv",
        "litgrid": PROCESSED_DIR / "litgrid_features_15min.csv",
        "flows": PROCESSED_DIR / "flows_15min.csv",
    },
}

PREDICTION_FILES = {
    "15min_clean": {
        "XGBoost": PROCESSED_DIR / "xgb_predictions_15min_clean.csv",
        "LightGBM": PROCESSED_DIR / "lgbm_predictions_15min_clean.csv",
        "CatBoost": PROCESSED_DIR / "catboost_predictions_15min_clean.csv",
        "MLP": PROCESSED_DIR / "mlp_predictions_15min_clean.csv",
        "LSTM": PROCESSED_DIR / "lstm_predictions_15min_clean.csv",
        "Ensemble": PROCESSED_DIR / "ensemble_predictions_15min_clean.csv",
    },
    "15min_extended": {
        "XGBoost": PROCESSED_DIR / "xgb_predictions_15min_extended.csv",
        "LightGBM": PROCESSED_DIR / "lgbm_predictions_15min_extended.csv",
        "CatBoost": PROCESSED_DIR / "catboost_predictions_15min_extended.csv",
        "MLP": PROCESSED_DIR / "mlp_predictions_15min_extended.csv",
        "LSTM": PROCESSED_DIR / "lstm_predictions_15min_extended.csv",
        "Ensemble": PROCESSED_DIR / "ensemble_predictions_15min_extended.csv",
    },
    "hourly_clean": {
        "XGBoost": PROCESSED_DIR / "xgb_predictions_hourly_clean.csv",
        "LightGBM": PROCESSED_DIR / "lgbm_predictions_hourly_clean.csv",
        "CatBoost": PROCESSED_DIR / "catboost_predictions_hourly_clean.csv",
        "MLP": PROCESSED_DIR / "mlp_predictions_hourly_clean.csv",
        "LSTM": PROCESSED_DIR / "lstm_predictions_hourly_clean.csv",
        "Ensemble": PROCESSED_DIR / "ensemble_predictions_hourly_clean.csv",
    },
    "hourly_extended": {
        "XGBoost": PROCESSED_DIR / "xgb_predictions_hourly_extended.csv",
        "LightGBM": PROCESSED_DIR / "lgbm_predictions_hourly_extended.csv",
        "CatBoost": PROCESSED_DIR / "catboost_predictions_hourly_extended.csv",
        "MLP": PROCESSED_DIR / "mlp_predictions_hourly_extended.csv",
        "LSTM": PROCESSED_DIR / "lstm_predictions_hourly_extended.csv",
        "Ensemble": PROCESSED_DIR / "ensemble_predictions_hourly_extended.csv",
    },
}

FEATURE_IMPORTANCE_FILES = {
    "15min_clean": {
        "XGBoost": REPORTS_DIR / "xgb_feature_importance_15min_clean.csv",
        "LightGBM": REPORTS_DIR / "lgbm_feature_importance_15min_clean.csv",
        "CatBoost": REPORTS_DIR / "catboost_feature_importance_15min_clean.csv",
    },
    "15min_extended": {
        "XGBoost": REPORTS_DIR / "xgb_feature_importance_15min_extended.csv",
        "LightGBM": REPORTS_DIR / "lgbm_feature_importance_15min_extended.csv",
        "CatBoost": REPORTS_DIR / "catboost_feature_importance_15min_extended.csv",
    },
    "hourly_clean": {
        "XGBoost": REPORTS_DIR / "xgb_feature_importance_hourly_clean.csv",
        "LightGBM": REPORTS_DIR / "lgbm_feature_importance_hourly_clean.csv",
        "CatBoost": REPORTS_DIR / "catboost_feature_importance_hourly_clean.csv",
    },
    "hourly_extended": {
        "XGBoost": REPORTS_DIR / "xgb_feature_importance_hourly_extended.csv",
        "LightGBM": REPORTS_DIR / "lgbm_feature_importance_hourly_extended.csv",
        "CatBoost": REPORTS_DIR / "catboost_feature_importance_hourly_extended.csv",
    },
}

METRICS_FILE = PROCESSED_DIR / "model_comparison.csv"


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"Nerastas failas: {path}")
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"Nepavyko nuskaityti failo {path}: {exc}")
        return None


def parse_datetime(df: pd.DataFrame, column: str = "datetime") -> pd.DataFrame:
    df = df.copy()
    df[column] = pd.to_datetime(df[column], errors="coerce")
    df = df.dropna(subset=[column]).sort_values(column).reset_index(drop=True)
    return df


def safe_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_or_create_data_version(session, version_name: str, description: str | None = None) -> DataVersion:
    version = session.query(DataVersion).filter_by(version_name=version_name).first()
    if version:
        return version

    version = DataVersion(
        version_name=version_name,
        description=description,
        created_at=datetime.now(UTC),
    )
    session.add(version)
    session.commit()
    session.refresh(version)
    return version


def delete_data_version(session, version_name: str) -> bool:
    version = session.query(DataVersion).filter_by(version_name=version_name).first()
    if not version:
        print(f"Nerasta data_version: {version_name}")
        return False

    session.delete(version)
    session.commit()
    print(f"✓ Ištrinta data_version: {version_name}")
    return True


def load_market_dataframe(frequency: str) -> pd.DataFrame | None:
    files = MARKET_FILES[frequency]

    df_master = safe_read_csv(files["master"])
    if df_master is None:
        return None

    df_master = parse_datetime(df_master, "datetime")
    df = df_master.copy()

    if files["litgrid"].exists():
        df_lit = safe_read_csv(files["litgrid"])
        if df_lit is not None:
            df_lit = parse_datetime(df_lit, "datetime")
            df = df.merge(df_lit, on="datetime", how="left")

    if files["flows"].exists():
        df_flows = safe_read_csv(files["flows"])
        if df_flows is not None:
            df_flows = parse_datetime(df_flows, "datetime")
            df = df.merge(df_flows, on="datetime", how="left")

    numeric_cols = [
        "price",
        "lv_price",
        "ee_price",
        "se4_price",
        "pl_price",
        "consumption_mw",
        "production_total_mw",
        "flow_lt_lv",
        "flow_lt_se",
        "flow_lt_pl",
        "flow_total",
        "flow_abs_total",
    ]
    df = safe_numeric(df, numeric_cols)

    df = (
        df.replace([float("inf"), float("-inf")], pd.NA)
        .sort_values("datetime")
        .drop_duplicates(subset=["datetime"], keep="last")
        .reset_index(drop=True)
    )
    return df


def upload_market_data(session, version: DataVersion):
    total_inserted = 0

    for frequency in ["hourly", "15min"]:
        df = load_market_dataframe(frequency)
        if df is None or df.empty:
            print(f"Praleidžiama market_data ({frequency}) – nėra duomenų")
            continue

        rows = []
        for _, row in df.iterrows():
            rows.append(
                MarketData(
                    data_version_id=version.id,
                    frequency=frequency,
                    datetime=row["datetime"],
                    price=row.get("price"),
                    lv_price=row.get("lv_price"),
                    ee_price=row.get("ee_price"),
                    se4_price=row.get("se4_price"),
                    pl_price=row.get("pl_price"),
                    consumption_mw=row.get("consumption_mw"),
                    production_total_mw=row.get("production_total_mw"),
                    flow_lt_lv=row.get("flow_lt_lv"),
                    flow_lt_se=row.get("flow_lt_se"),
                    flow_lt_pl=row.get("flow_lt_pl"),
                    flow_total=row.get("flow_total"),
                    flow_abs_total=row.get("flow_abs_total"),
                )
            )

        if rows:
            session.bulk_save_objects(rows)
            session.commit()
            total_inserted += len(rows)

        print(f"✓ Sukelta market_data ({frequency}): {len(rows)} eilučių")

    print(f"✓ Iš viso market_data eilučių: {total_inserted}")


def upload_predictions(session, version: DataVersion):
    total_rows = 0

    for dataset_name, models in PREDICTION_FILES.items():
        for model_name, path in models.items():
            df = safe_read_csv(path)
            if df is None:
                continue

            required_cols = {"datetime", "price", "predicted_price"}
            if not required_cols.issubset(df.columns):
                print(f"Praleista predictions dėl blogo formato: {path}")
                continue

            df = parse_datetime(df, "datetime")
            df = safe_numeric(df, ["price", "predicted_price"])

            df["abs_error"] = (df["predicted_price"] - df["price"]).abs()
            df["error"] = df["predicted_price"] - df["price"]

            df = df.dropna(subset=["datetime", "price", "predicted_price"]).reset_index(drop=True)

            rows = []
            for _, row in df.iterrows():
                rows.append(
                    Prediction(
                        data_version_id=version.id,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        datetime=row["datetime"],
                        actual_price=row.get("price"),
                        predicted_price=row.get("predicted_price"),
                        abs_error=row.get("abs_error"),
                        error=row.get("error"),
                    )
                )

            if rows:
                session.bulk_save_objects(rows)
                session.commit()
                total_rows += len(rows)

            print(f"✓ Sukelta predictions: {dataset_name} / {model_name} / {len(rows)} eilučių")

    print(f"✓ Iš viso prediction eilučių: {total_rows}")


def upload_metrics(session, version: DataVersion):
    df = safe_read_csv(METRICS_FILE)
    if df is None:
        print("Praleista model_metrics – nerastas model_comparison.csv")
        return

    required = {"data", "model", "mae", "rmse", "r2", "smape"}
    if not required.issubset(df.columns):
        print("Praleista model_metrics – blogas model_comparison.csv formatas")
        return

    df = safe_numeric(df, ["mae", "rmse", "r2", "smape"])
    df = df.dropna(subset=["data", "model", "mae", "rmse", "r2", "smape"]).reset_index(drop=True)

    rows = []
    for _, row in df.iterrows():
        rows.append(
            ModelMetric(
                data_version_id=version.id,
                dataset_name=row["data"],
                model_name=row["model"],
                mae=row["mae"],
                rmse=row["rmse"],
                r2=row["r2"],
                smape=row["smape"],
            )
        )

    if rows:
        session.bulk_save_objects(rows)
        session.commit()

    print(f"✓ Sukelta model_metrics: {len(rows)} eilučių")


def upload_feature_importance(session, version: DataVersion):
    total_rows = 0

    for dataset_name, models in FEATURE_IMPORTANCE_FILES.items():
        for model_name, path in models.items():
            df = safe_read_csv(path)
            if df is None:
                continue

            if "feature" not in df.columns:
                print(f"Praleista feature importance dėl blogo formato: {path}")
                continue

            importance_col = None
            for candidate in ["importance", "importance_gain", "abs_coefficient", "coefficient"]:
                if candidate in df.columns:
                    importance_col = candidate
                    break

            if importance_col is None:
                print(f"Praleista feature importance – nerastas importance stulpelis: {path}")
                continue

            df = safe_numeric(df, [importance_col])
            df = df.dropna(subset=["feature", importance_col]).reset_index(drop=True)

            rows = []
            for _, row in df.iterrows():
                rows.append(
                    FeatureImportance(
                        data_version_id=version.id,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        feature=row["feature"],
                        importance=row[importance_col],
                    )
                )

            if rows:
                session.bulk_save_objects(rows)
                session.commit()
                total_rows += len(rows)

            print(f"✓ Sukelta feature_importance: {dataset_name} / {model_name} / {len(rows)} eilučių")

    print(f"✓ Iš viso feature_importance eilučių: {total_rows}")


def init_db():
    Base.metadata.create_all(bind=engine)
    print("✓ DB schema sukurta / atnaujinta")


def load_version(version_name: str, description: str | None, replace: bool):
    init_db()

    session = SessionLocal()
    try:
        if replace:
            delete_data_version(session, version_name)

        version = get_or_create_data_version(session, version_name, description)

        existing_market = session.query(MarketData).filter_by(data_version_id=version.id).first()
        if existing_market and not replace:
            raise ValueError(
                f"Data version '{version_name}' jau turi duomenis. "
                f"Naudok --replace arba kitą --version-name."
            )

        upload_market_data(session, version)
        upload_predictions(session, version)
        upload_metrics(session, version)
        upload_feature_importance(session, version)

        print(f"\n✓ Baigta. Data version: {version_name}")
    finally:
        session.close()


def list_versions():
    init_db()
    session = SessionLocal()
    try:
        versions = session.query(DataVersion).order_by(DataVersion.id.asc()).all()
        if not versions:
            print("DB dar nėra data_versions.")
            return

        print("\nEsamos data_versions:")
        for v in versions:
            print(f"- {v.version_name} | created_at={v.created_at} | description={v.description}")
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="Sukuria DB ir sukelia processed duomenis į ją")
    parser.add_argument("--version-name", type=str, help="Duomenų versijos pavadinimas, pvz. baseline_2026_03_15")
    parser.add_argument("--description", type=str, default=None, help="Papildomas versijos aprašymas")
    parser.add_argument("--replace", action="store_true", help="Jei versija egzistuoja, ištrinti ir įkelti iš naujo")
    parser.add_argument("--delete-version", type=str, default=None, help="Ištrina konkrečią data_version")
    parser.add_argument("--list-versions", action="store_true", help="Parodo visas data_version")
    args = parser.parse_args()

    if args.list_versions:
        list_versions()
        return

    if args.delete_version:
        init_db()
        session = SessionLocal()
        try:
            delete_data_version(session, args.delete_version)
        finally:
            session.close()
        return

    if not args.version_name:
        raise ValueError("Nurodyk --version-name, pvz. baseline_2026_03_15")

    load_version(
        version_name=args.version_name,
        description=args.description,
        replace=args.replace,
    )


if __name__ == "__main__":
    main()