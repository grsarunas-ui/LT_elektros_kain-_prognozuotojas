from app.db import engine
from sqlalchemy import text

VERSION_NAME = "Forecast2026_03_01-2026-03-15"

with engine.begin() as conn:
    # 1. Gaunam ID
    result = conn.execute(
        text("""
            SELECT id FROM data_versions
            WHERE version_name = :version_name
        """),
        {"version_name": VERSION_NAME}
    ).fetchone()

    if not result:
        print(f"❌ Nerastas version_name: {VERSION_NAME}")
        exit()

    data_version_id = result[0]
    print(f"🔍 Found data_version_id: {data_version_id}")

    # 2. Trinam iš visų dependent lentelių
    tables = [
        "feature_importance",
        "market_data",
        "model_metrics",
        "predictions",
    ]

    for table in tables:
        deleted = conn.execute(
            text(f"DELETE FROM {table} WHERE data_version_id = :id"),
            {"id": data_version_id}
        )
        print(f"🧹 {table}: deleted {deleted.rowcount} rows")

    # 3. Trinam pagrindinę lentelę
    deleted = conn.execute(
        text("""
            DELETE FROM data_versions
            WHERE id = :id
        """),
        {"id": data_version_id}
    )

    print(f"🧨 data_versions: deleted {deleted.rowcount} rows")

print("✅ DONE")