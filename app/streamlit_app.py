import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
import streamlit as st
from sqlalchemy import text

from app.db import engine

try:
    from app.training_utils import train_xgboost_interactive
except ImportError:
    train_xgboost_interactive = None

try:
    from app.training_utils import train_lightgbm_interactive
except ImportError:
    train_lightgbm_interactive = None

try:
    from app.training_utils import train_catboost_interactive
except ImportError:
    train_catboost_interactive = None

try:
    from app.training_utils import train_mlp_interactive
except ImportError:
    train_mlp_interactive = None

try:
    from app.training_utils import train_lstm_interactive
except ImportError:
    train_lstm_interactive = None


st.set_page_config(
    page_title="LT elektros kainų prognozė",
    page_icon="⚡",
    layout="wide",
)

WEATHER_FILE = Path("data/processed/weather_lithuania_avg_hourly.csv")


@st.cache_data
def run_query(query: str, params: dict | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)


@st.cache_data
def get_data_versions() -> pd.DataFrame:
    query = """
        SELECT id, version_name, description, created_at
        FROM data_versions
        ORDER BY id
    """
    return run_query(query)


@st.cache_data
def get_model_metrics(version_name: str) -> pd.DataFrame:
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
    return run_query(query, {"version_name": version_name})


@st.cache_data
def get_predictions(version_name: str, dataset_name: str, model_name: str) -> pd.DataFrame:
    query = """
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
    """
    df = run_query(
        query,
        {
            "version_name": version_name,
            "dataset_name": dataset_name,
            "model_name": model_name,
        },
    )
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


@st.cache_data
def get_feature_importance(version_name: str, dataset_name: str, model_name: str) -> pd.DataFrame:
    query = """
        SELECT
            fi.feature,
            fi.importance
        FROM feature_importance fi
        JOIN data_versions dv ON fi.data_version_id = dv.id
        WHERE dv.version_name = :version_name
          AND fi.dataset_name = :dataset_name
          AND fi.model_name = :model_name
        ORDER BY fi.importance DESC
    """
    return run_query(
        query,
        {
            "version_name": version_name,
            "dataset_name": dataset_name,
            "model_name": model_name,
        },
    )


@st.cache_data
def get_market_data(version_name: str, frequency: str = "hourly") -> pd.DataFrame:
    query = """
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
    """
    df = run_query(
        query,
        {
            "version_name": version_name,
            "frequency": frequency,
        },
    )
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"] = df["datetime"].dt.date
        df["hour"] = df["datetime"].dt.hour
    return df


@st.cache_data
def get_weather_data() -> pd.DataFrame:
    if not WEATHER_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(WEATHER_FILE)
    if "datetime" not in df.columns:
        return pd.DataFrame()

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    return df


def make_metric_cards(metrics_df: pd.DataFrame):
    if metrics_df.empty:
        st.warning("Nėra modelių metrikų šiai versijai.")
        return

    best_row = metrics_df.iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Geriausias modelis", f"{best_row['model_name']} / {best_row['dataset_name']}")
    c2.metric("MAE", f"{best_row['mae']:.2f}")
    c3.metric("RMSE", f"{best_row['rmse']:.2f}")
    c4.metric("R²", f"{best_row['r2']:.4f}")
    c5.metric("sMAPE", f"{best_row['smape']:.2f}%")


def downsample_df(df: pd.DataFrame, max_points: int = 2000) -> pd.DataFrame:
    if df.empty or len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()


def show_header(selected_version_row: pd.Series):
    st.title("⚡ Lietuvos elektros kainų prognozavimo sistema")
    c1, c2, c3 = st.columns([2, 1, 2])
    c1.markdown(f"**Duomenų versija:** `{selected_version_row['version_name']}`")
    c2.markdown(f"**Sukurta:** `{selected_version_row['created_at']}`")
    c3.markdown(f"**Aprašymas:** {selected_version_row['description']}")


def get_interactive_trainers():
    trainers = {}

    if train_xgboost_interactive is not None:
        trainers["XGBoost"] = train_xgboost_interactive
    if train_lightgbm_interactive is not None:
        trainers["LightGBM"] = train_lightgbm_interactive
    if train_catboost_interactive is not None:
        trainers["CatBoost"] = train_catboost_interactive
    if train_mlp_interactive is not None:
        trainers["MLP"] = train_mlp_interactive
    if train_lstm_interactive is not None:
        trainers["LSTM"] = train_lstm_interactive

    return trainers


def render_prediction_scatter(pred_df: pd.DataFrame):
    scatter_df = pred_df[["actual_price", "predicted_price", "abs_error", "datetime"]].copy()
    scatter_df = scatter_df.rename(
        columns={
            "actual_price": "Faktinė kaina",
            "predicted_price": "Prognozuota kaina",
            "abs_error": "Absoliuti klaida",
            "datetime": "Data laikas",
        }
    )

    min_price = float(
        min(scatter_df["Faktinė kaina"].min(), scatter_df["Prognozuota kaina"].min())
    )
    max_price = float(
        max(scatter_df["Faktinė kaina"].max(), scatter_df["Prognozuota kaina"].max())
    )

    diagonal_df = pd.DataFrame({
        "Faktinė kaina": [min_price, max_price],
        "Ideali prognozė": [min_price, max_price],
    })

    scatter_points = (
        alt.Chart(scatter_df)
        .mark_circle(size=60, opacity=0.55)
        .encode(
            x=alt.X("Faktinė kaina:Q", title="Faktinė kaina"),
            y=alt.Y("Prognozuota kaina:Q", title="Prognozuota kaina"),
            tooltip=[
                alt.Tooltip("Data laikas:T", title="Data laikas"),
                alt.Tooltip("Faktinė kaina:Q", format=".2f"),
                alt.Tooltip("Prognozuota kaina:Q", format=".2f"),
                alt.Tooltip("Absoliuti klaida:Q", format=".2f"),
            ],
        )
    )

    diagonal_line = (
        alt.Chart(diagonal_df)
        .mark_line()
        .encode(
            x=alt.X("Faktinė kaina:Q"),
            y=alt.Y("Ideali prognozė:Q"),
        )
    )

    st.markdown("### Prognozių išsibarstymas nuo realių reikšmių")
    st.altair_chart((diagonal_line + scatter_points).properties(height=420), use_container_width=True)


def render_residual_scatter(pred_df: pd.DataFrame):
    residual_df = pred_df[["actual_price", "error", "abs_error", "datetime"]].copy()
    residual_df = residual_df.rename(
        columns={
            "actual_price": "Faktinė kaina",
            "error": "Paklaida",
            "abs_error": "Absoliuti klaida",
            "datetime": "Data laikas",
        }
    )

    zero_line_df = pd.DataFrame({
        "Faktinė kaina": [
            float(residual_df["Faktinė kaina"].min()),
            float(residual_df["Faktinė kaina"].max()),
        ],
        "Nulinė paklaida": [0.0, 0.0],
    })

    residual_points = (
        alt.Chart(residual_df)
        .mark_circle(size=60, opacity=0.55)
        .encode(
            x=alt.X("Faktinė kaina:Q", title="Faktinė kaina"),
            y=alt.Y("Paklaida:Q", title="Paklaida (prognozė - faktinė)"),
            tooltip=[
                alt.Tooltip("Data laikas:T", title="Data laikas"),
                alt.Tooltip("Faktinė kaina:Q", format=".2f"),
                alt.Tooltip("Paklaida:Q", format=".2f"),
                alt.Tooltip("Absoliuti klaida:Q", format=".2f"),
            ],
        )
    )

    zero_line = (
        alt.Chart(zero_line_df)
        .mark_line()
        .encode(
            x=alt.X("Faktinė kaina:Q"),
            y=alt.Y("Nulinė paklaida:Q"),
        )
    )

    st.markdown("### Residual scatter grafikas")
    st.altair_chart((zero_line + residual_points).properties(height=360), use_container_width=True)


def render_prediction_analysis(
    pred_df: pd.DataFrame,
    importance_df: pd.DataFrame | None = None,
    top_errors_df: pd.DataFrame | None = None,
):
    st.markdown("### Faktinė vs prognozuota kaina laike")
    st.line_chart(pred_df.set_index("datetime")[["actual_price", "predicted_price"]], height=420)

    st.markdown("### Absoliuti klaida laike")
    st.line_chart(pred_df.set_index("datetime")[["abs_error"]], height=260)

    render_prediction_scatter(pred_df)
    render_residual_scatter(pred_df)

    tabs = st.tabs(["Top features", "Didžiausios klaidos", "Paskutinės prognozės"])

    with tabs[0]:
        if importance_df is not None and not importance_df.empty:
            st.dataframe(importance_df.head(20), use_container_width=True)

            chart_df = importance_df.head(20).copy()
            importance_col = None
            for col in ["importance", "importance_gain", "abs_coefficient", "coefficient"]:
                if col in chart_df.columns:
                    importance_col = col
                    break

            if importance_col is not None:
                st.markdown("### Feature importance grafikas")
                st.bar_chart(chart_df.set_index("feature")[importance_col])
            else:
                st.info("Feature importance grafiko nepavyko parodyti, nes nerastas tinkamas stulpelis.")
        else:
            st.info("Šiam modeliui feature importance duomenų nerasta.")

    with tabs[1]:
        if top_errors_df is not None and not top_errors_df.empty:
            st.dataframe(top_errors_df.head(20), use_container_width=True)
        else:
            st.dataframe(pred_df.sort_values("abs_error", ascending=False).head(20), use_container_width=True)

    with tabs[2]:
        st.dataframe(pred_df.tail(50), use_container_width=True)


def show_project_info():
    st.subheader("📘 Project Info")

    st.markdown("""
### Projekto tikslas
Sukurti naudotojui draugišką sistemą, kuri:
- analizuoja Lietuvos elektros kainų duomenis,
- leidžia palyginti kelis modelius,
- leidžia interaktyviai treniruoti modelį su pasirenkamais train/test intervalais,
- naudoja duomenų bazę vietoje vien tik CSV failų.

### Naudojami modeliai
- **XGBoost**
- **LightGBM**
- **CatBoost**
- **MLP**
- **LSTM**
- **Ensemble**

### Pagrindinės metrikos
- **MAE**
- **RMSE**
- **R²**
- **sMAPE**
""")


def show_dashboard(metrics_df: pd.DataFrame):
    st.subheader("🏠 Dashboard")
    make_metric_cards(metrics_df)

    if metrics_df.empty:
        return

    best_row = metrics_df.iloc[0]
    st.success(
        f"Geriausias dabartinės versijos modelis: **{best_row['model_name']} / {best_row['dataset_name']}**, "
        f"MAE = **{best_row['mae']:.2f}**, RMSE = **{best_row['rmse']:.2f}**, "
        f"R² = **{best_row['r2']:.4f}**, sMAPE = **{best_row['smape']:.2f}%**"
    )

    st.markdown("### Modelių metrikos")
    st.dataframe(metrics_df, use_container_width=True)


def show_data_explorer(selected_version: str):
    st.subheader("📊 Data Explorer")
    freq_choice = st.radio("Pasirink dažnį", ["hourly", "15min"], horizontal=True)

    market_df = get_market_data(selected_version, freq_choice)
    if market_df.empty:
        st.warning("Nėra rinkos duomenų pasirinktam dažniui.")
        return

    weather_df = get_weather_data()

    min_dt = market_df["datetime"].min()
    max_dt = market_df["datetime"].max()

    c1, c2, c3 = st.columns(3)
    date_range = c1.date_input(
        "Laikotarpis",
        value=(min_dt.date(), max_dt.date()),
        min_value=min_dt.date(),
        max_value=max_dt.date(),
    )
    max_chart_points = c2.selectbox("Maks. taškų grafike", [500, 1000, 2000, 5000], index=1)
    table_rows = c3.selectbox("Eilučių lentelėje", [20, 50, 100, 200], index=1)

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_dt.date(), max_dt.date()

    filtered_df = market_df[
        (market_df["datetime"].dt.date >= start_date) &
        (market_df["datetime"].dt.date <= end_date)
    ].copy()

    if filtered_df.empty:
        st.warning("Pagal pasirinktą laikotarpį duomenų nėra.")
        return

    chart_df = downsample_df(filtered_df, max_points=max_chart_points)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Eilučių", len(filtered_df))
    c2.metric("Pradžia", str(filtered_df["datetime"].min()))
    c3.metric("Pabaiga", str(filtered_df["datetime"].max()))
    c4.metric("Vid. LT kaina", f"{filtered_df['price'].mean():.2f}")

    tabs = st.tabs([
        "Nord Pool kainos",
        "Litgrid duomenys",
        "Commercial flows",
        "Orų duomenys",
        "Duomenų lentelė",
    ])

    with tabs[0]:
        st.markdown("### Nord Pool zonų kainos")
        price_cols = ["price", "lv_price", "ee_price", "se4_price", "pl_price"]
        existing_price_cols = [c for c in price_cols if c in chart_df.columns]

        if existing_price_cols:
            st.line_chart(chart_df.set_index("datetime")[existing_price_cols], height=420)

        st.dataframe(filtered_df[["datetime"] + existing_price_cols].tail(table_rows), use_container_width=True)

    with tabs[1]:
        litgrid_cols = [c for c in ["consumption_mw", "production_total_mw"] if c in chart_df.columns]
        if litgrid_cols:
            st.line_chart(chart_df.set_index("datetime")[litgrid_cols], height=360)

    with tabs[2]:
        flow_cols = [c for c in ["flow_lt_lv", "flow_lt_se", "flow_lt_pl", "flow_total", "flow_abs_total"] if c in chart_df.columns]
        if flow_cols:
            st.line_chart(chart_df.set_index("datetime")[flow_cols], height=380)

    with tabs[3]:
        if weather_df.empty:
            st.info("Weather failas nerastas arba jame nėra duomenų.")
        else:
            weather_filtered = weather_df[
                (weather_df["datetime"] >= filtered_df["datetime"].min()) &
                (weather_df["datetime"] <= filtered_df["datetime"].max())
            ].copy()
            weather_chart = downsample_df(weather_filtered, max_points=max_chart_points)
            weather_cols = [
                c for c in [
                    "temperature_2m",
                    "wind_speed_10m",
                    "cloud_cover",
                    "shortwave_radiation",
                    "solar_proxy",
                ] if c in weather_chart.columns
            ]
            if weather_cols:
                st.line_chart(weather_chart.set_index("datetime")[weather_cols], height=420)

    with tabs[4]:
        st.dataframe(filtered_df.tail(table_rows), use_container_width=True)


def show_model_analysis(selected_version: str, metrics_df: pd.DataFrame):
    st.subheader("🤖 Model Analysis")

    if metrics_df.empty:
        st.warning("Nėra modelių metrikų šiai versijai.")
        return

    c1, c2 = st.columns(2)

    dataset_options = metrics_df["dataset_name"].drop_duplicates().tolist()
    model_options = metrics_df["model_name"].drop_duplicates().tolist()

    default_dataset = metrics_df.iloc[0]["dataset_name"]
    default_model = metrics_df.iloc[0]["model_name"]

    selected_dataset = c1.selectbox("Pasirink dataset", dataset_options, index=dataset_options.index(default_dataset))
    selected_model = c2.selectbox("Pasirink modelį", model_options, index=model_options.index(default_model))

    pred_df = get_predictions(selected_version, selected_dataset, selected_model)
    importance_df = get_feature_importance(selected_version, selected_dataset, selected_model)

    if pred_df.empty:
        st.warning("Nėra prognozių pasirinktam deriniui.")
        return

    y_true = pred_df["actual_price"]
    y_pred = pred_df["predicted_price"]

    mae = float(pred_df["abs_error"].mean())
    rmse = float((((y_pred - y_true) ** 2).mean()) ** 0.5)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    smape = float((2 * (y_pred - y_true).abs() / (y_true.abs() + y_pred.abs() + 1e-6)).mean() * 100)
    max_abs_error = float(pred_df["abs_error"].max())

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("MAE (visas testas)", f"{mae:.2f}")
    m2.metric("RMSE (visas testas)", f"{rmse:.2f}")
    m3.metric("R² (visas testas)", f"{r2:.4f}")
    m4.metric("sMAPE (visas testas)", f"{smape:.2f}%")
    m5.metric("Max abs error (visas testas)", f"{max_abs_error:.2f}")

    render_prediction_analysis(pred_df, importance_df=importance_df)


def show_data_versions(versions_df: pd.DataFrame, metrics_df: pd.DataFrame):
    st.subheader("🗄️ Data Versions")
    st.dataframe(versions_df, use_container_width=True)
    st.dataframe(metrics_df, use_container_width=True)


def show_interactive_training():
    st.subheader("🧪 Interactive Training")

    trainers = get_interactive_trainers()
    if not trainers:
        st.error("Nerasta nė vienos interactive training funkcijos `app.training_utils` faile.")
        return

    c0, c1 = st.columns(2)
    model_name = c0.selectbox("Modelis", options=list(trainers.keys()), index=0)
    dataset_name = c1.selectbox(
        "Dataset",
        options=["15min_clean", "15min_extended", "hourly_clean", "hourly_extended"],
        index=3,
    )

    c2, c3 = st.columns(2)
    train_start = c2.date_input("Train nuo", value=pd.to_datetime("2024-01-01").date())
    train_end = c3.date_input("Train iki", value=pd.to_datetime("2026-03-15").date())

    c4, c5, c6 = st.columns([1, 1, 1])
    test_start = c4.date_input("Test nuo", value=pd.to_datetime("2026-03-16").date())
    horizon_choice = c5.selectbox("Test horizontas", ["1 diena", "3 dienos", "7 dienos", "14 dienų", "Custom"], index=3)

    if horizon_choice == "Custom":
        test_end = c6.date_input("Test iki", value=pd.to_datetime("2026-03-31").date())
    else:
        horizon_map = {"1 diena": 1, "3 dienos": 3, "7 dienos": 7, "14 dienų": 14}
        test_end = pd.to_datetime(test_start) + pd.Timedelta(days=horizon_map[horizon_choice])
        c6.markdown("**Test iki**")
        c6.code(str(test_end.date()))

    run_name = st.text_input(
        "Run name",
        value=f"{model_name.lower()}_{dataset_name}_{train_start}_{test_start}_{test_end if isinstance(test_end, pd.Timestamp) else test_end}",
    )

    if st.button("Paleisti modelio treniravimą", use_container_width=True):
        trainer = trainers[model_name]

        with st.spinner(f"Treniruojamas modelis: {model_name}..."):
            try:
                result = trainer(
                    dataset_name=dataset_name,
                    train_start=str(train_start),
                    train_end=str(train_end),
                    test_start=str(test_start),
                    test_end=str(test_end.date() if hasattr(test_end, "date") else test_end),
                    run_name=run_name,
                )

                summary = result["summary"]
                pred_df = result["predictions"].copy()
                importance_df = result.get("importance", pd.DataFrame())
                top_errors_df = result.get("top_errors", pd.DataFrame())

                if "datetime" in pred_df.columns:
                    pred_df["datetime"] = pd.to_datetime(pred_df["datetime"])

                if "actual_price" not in pred_df.columns and "price" in pred_df.columns:
                    pred_df["actual_price"] = pred_df["price"]

                if "error" not in pred_df.columns and {"predicted_price", "actual_price"}.issubset(pred_df.columns):
                    pred_df["error"] = pred_df["predicted_price"] - pred_df["actual_price"]

                if "abs_error" not in pred_df.columns and "error" in pred_df.columns:
                    pred_df["abs_error"] = pred_df["error"].abs()

                max_abs_error = float(pred_df["abs_error"].max())

                st.success(f"Treniravimas baigtas. Rezultatai išsaugoti: {result['run_dir']}")

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("MAE", f"{summary['mae']:.2f}")
                c2.metric("RMSE", f"{summary['rmse']:.2f}")
                c3.metric("R²", f"{summary['r2']:.4f}")
                c4.metric("sMAPE", f"{summary['smape']:.2f}%")
                c5.metric("Max abs error (visas testas)", f"{max_abs_error:.2f}")

                render_prediction_analysis(pred_df, importance_df=importance_df, top_errors_df=top_errors_df)

            except Exception as e:
                st.error(f"Klaida treniruojant modelį: {e}")


def main():
    versions_df = get_data_versions()
    if versions_df.empty:
        st.error("Duomenų bazėje nerasta nė vienos data_version. Pirma paleisk load_to_db.")
        return

    selected_version = st.selectbox(
        "Pasirink duomenų versiją",
        options=versions_df["version_name"].tolist(),
        index=len(versions_df) - 1,
    )

    selected_version_row = versions_df[versions_df["version_name"] == selected_version].iloc[0]
    show_header(selected_version_row)

    nav_options = [
        "Project Info",
        "Dashboard",
        "Data Explorer",
        "Model Analysis",
        "Interactive Training",
        "Data Versions",
    ]

    page = st.radio("Navigacija", options=nav_options, horizontal=True)

    st.divider()

    metrics_df = get_model_metrics(selected_version)

    if page == "Project Info":
        show_project_info()
    elif page == "Dashboard":
        show_dashboard(metrics_df)
    elif page == "Data Explorer":
        show_data_explorer(selected_version)
    elif page == "Model Analysis":
        show_model_analysis(selected_version, metrics_df)
    elif page == "Interactive Training":
        show_interactive_training()
    elif page == "Data Versions":
        show_data_versions(versions_df, metrics_df)


if __name__ == "__main__":
    main()