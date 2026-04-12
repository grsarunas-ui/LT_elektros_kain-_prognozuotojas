from __future__ import annotations

import argparse
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd

from app.db import Base, SessionLocal, engine
from app.models import DataVersion, MarketData, Prediction, ModelMetric


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


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
        "MLP": PROCESSED_DIR / "mlp_predictions_15min_clean.csv",
    },
    "15min_extended": {
        "XGBoost": PROCESSED_DIR / "xgb_predictions_15min_extended.csv",
        "MLP": PROCESSED_DIR / "mlp_predictions_15min_extended.csv",
    },
    "hourly_clean": {
        "XGBoost": PROCESSED_DIR / "xgb_predictions_hourly_clean.csv",
        "MLP": PROCESSED_DIR / "mlp_predictions_hourly_clean.csv",
    },
    "hourly_extended": {
        "XGBoost": PROCESSED_DIR / "xgb_predictions_hourly_extended.csv",
        "MLP": PROCESSED_DIR / "mlp_predictions_hourly_extended.csv",
    },
}

METRICS_FILE = PROCESSED_DIR / "model_comparison.csv"


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"Nerastas failas: {path}")
        return None
    return pd.read_csv(path)


def parse_datetime(df: pd.DataFrame, column: str = "datetime") -> pd.DataFrame:
    df = df.copy()
    df[column] = pd.to_datetime(df[column], errors="coerce")
    df = df.dropna(subset=[column]).sort_values(column).reset_index(drop=True)
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

    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)
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

            if not {"datetime", "price", "predicted_price"}.issubset(df.columns):
                print(f"Praleista predictions dėl blogo formato: {path}")
                continue

            df = parse_datetime(df, "datetime")
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["predicted_price"] = pd.to_numeric(df["predicted_price"], errors="coerce")
            df["abs_error"] = (df["predicted_price"] - df["price"]).abs()
            df["error"] = df["predicted_price"] - df["price"]

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

    session.bulk_save_objects(rows)
    session.commit()
    print(f"✓ Sukelta model_metrics: {len(rows)} eilučių")


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