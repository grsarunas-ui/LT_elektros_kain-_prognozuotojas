import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sqlalchemy import text

from app.db import engine


st.set_page_config(
    page_title="LT elektros kainų prognozė",
    page_icon="⚡",
    layout="wide",
)


# =========================
# DB HELPERS
# =========================
@st.cache_data
def run_query(query: str, params: dict | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params)


@st.cache_data
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
    df = run_query(query, {
        "version_name": version_name,
        "dataset_name": dataset_name,
        "model_name": model_name,
    })

    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


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
    df = run_query(query, {
        "version_name": version_name,
        "frequency": frequency,
    })

    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["date"] = df["datetime"].dt.date
        df["hour"] = df["datetime"].dt.hour
        df["weekday"] = df["datetime"].dt.weekday
    return df


# =========================
# UI HELPERS
# =========================
def make_metric_cards(metrics_df: pd.DataFrame):
    if metrics_df.empty:
        st.warning("Nėra modelių metrikų šiai versijai.")
        return

    best_row = metrics_df.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Geriausias modelis", f"{best_row['model_name']} / {best_row['dataset_name']}")
    c2.metric("MAE", f"{best_row['mae']:.2f}")
    c3.metric("RMSE", f"{best_row['rmse']:.2f}")
    c4.metric("R²", f"{best_row['r2']:.4f}")


def downsample_df(df: pd.DataFrame, max_points: int = 2000) -> pd.DataFrame:
    if df.empty or len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()


def make_heatmap(df: pd.DataFrame, value_col: str, title: str):
    if df.empty or value_col not in df.columns:
        st.info(f"Nėra duomenų heatmap grafikui: {value_col}")
        return

    pivot = df.pivot_table(
        index="date",
        columns="hour",
        values=value_col,
        aggfunc="mean"
    )

    if pivot.empty:
        st.info(f"Nėra duomenų heatmap grafikui: {value_col}")
        return

    fig_height = min(12, max(4, len(pivot) * 0.18))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("Valanda")
    ax.set_ylabel("Data")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)

    y_step = max(1, len(pivot.index) // 12)
    ax.set_yticks(range(0, len(pivot.index), y_step))
    ax.set_yticklabels([str(pivot.index[i]) for i in range(0, len(pivot.index), y_step)])

    fig.colorbar(im, ax=ax)
    st.pyplot(fig)
    plt.close(fig)


# =========================
# PAGE: DASHBOARD
# =========================
def show_dashboard(selected_version: str, metrics_df: pd.DataFrame):
    st.subheader("🏠 Dashboard")
    make_metric_cards(metrics_df)

    st.markdown("### Modelių metrikos")
    st.dataframe(metrics_df, use_container_width=True)

    if not metrics_df.empty:
        best_row = metrics_df.iloc[0]
        st.success(
            f"Geriausias modelis šiai versijai yra **{best_row['model_name']} / {best_row['dataset_name']}**, "
            f"su **MAE {best_row['mae']:.2f}**, **RMSE {best_row['rmse']:.2f}**, **R² {best_row['r2']:.4f}**."
        )


# =========================
# PAGE: DATA EXPLORER
# =========================
def show_data_explorer(selected_version: str):
    st.subheader("📊 Data Explorer")

    freq_choice = st.radio(
        "Pasirink dažnį",
        options=["hourly", "15min"],
        horizontal=True,
    )

    market_df = get_market_data(selected_version, freq_choice)

    if market_df.empty:
        st.warning("Nėra rinkos duomenų pasirinktam dažniui.")
        return

    st.markdown("### Filtrai")

    min_dt = market_df["datetime"].min()
    max_dt = market_df["datetime"].max()

    c1, c2, c3 = st.columns(3)

    date_range = c1.date_input(
        "Laikotarpis",
        value=(min_dt.date(), max_dt.date()),
        min_value=min_dt.date(),
        max_value=max_dt.date(),
    )

    max_chart_points = c2.selectbox(
        "Maks. taškų grafike",
        options=[500, 1000, 2000, 5000],
        index=1,
    )

    table_rows = c3.selectbox(
        "Eilučių lentelėje",
        options=[20, 50, 100, 200],
        index=1,
    )

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

    subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs([
        "Nord Pool kainos",
        "Litgrid duomenys",
        "Commercial flows",
        "Heatmaps",
        "Duomenų lentelė",
    ])

    with subtab1:
        st.markdown("### Nord Pool zonų kainos")
        price_cols = ["price", "lv_price", "ee_price", "se4_price", "pl_price"]
        existing_price_cols = [c for c in price_cols if c in chart_df.columns]

        if existing_price_cols:
            st.line_chart(
                chart_df.set_index("datetime")[existing_price_cols],
                height=420
            )

        st.markdown("### LT spread'ai prieš kitas zonas")
        spread_df = chart_df[["datetime"]].copy()

        if "lv_price" in chart_df.columns:
            spread_df["spread_lv"] = chart_df["price"] - chart_df["lv_price"]
        if "ee_price" in chart_df.columns:
            spread_df["spread_ee"] = chart_df["price"] - chart_df["ee_price"]
        if "se4_price" in chart_df.columns:
            spread_df["spread_se4"] = chart_df["price"] - chart_df["se4_price"]
        if "pl_price" in chart_df.columns:
            spread_df["spread_pl"] = chart_df["price"] - chart_df["pl_price"]

        spread_cols = [c for c in spread_df.columns if c != "datetime"]
        if spread_cols:
            st.line_chart(
                spread_df.set_index("datetime")[spread_cols],
                height=320
            )

        st.markdown("### Paskutinės eilutės")
        st.dataframe(
            filtered_df[["datetime"] + existing_price_cols].tail(table_rows),
            use_container_width=True
        )

    with subtab2:
        st.markdown("### Litgrid vartojimas ir gamyba")

        litgrid_cols = [c for c in ["consumption_mw", "production_total_mw"] if c in chart_df.columns]
        if litgrid_cols:
            st.line_chart(
                chart_df.set_index("datetime")[litgrid_cols],
                height=360
            )

        if {"consumption_mw", "production_total_mw"}.issubset(filtered_df.columns):
            derived = filtered_df[["datetime"]].copy()
            derived["net_load"] = filtered_df["consumption_mw"] - filtered_df["production_total_mw"]
            derived["production_to_consumption"] = (
                filtered_df["production_total_mw"] / (filtered_df["consumption_mw"] + 1e-6)
            )

            derived_chart = downsample_df(derived, max_points=max_chart_points)

            st.markdown("### Net load")
            st.line_chart(
                derived_chart.set_index("datetime")[["net_load"]],
                height=250
            )

            st.markdown("### Production / Consumption santykis")
            st.line_chart(
                derived_chart.set_index("datetime")[["production_to_consumption"]],
                height=250
            )

        display_cols = [c for c in ["datetime", "consumption_mw", "production_total_mw"] if c in filtered_df.columns]
        st.dataframe(filtered_df[display_cols].tail(table_rows), use_container_width=True)

    with subtab3:
        st.markdown("### Commercial flows")

        flow_cols = [c for c in ["flow_lt_lv", "flow_lt_se", "flow_lt_pl", "flow_total", "flow_abs_total"] if c in chart_df.columns]
        if flow_cols:
            st.line_chart(
                chart_df.set_index("datetime")[flow_cols],
                height=380
            )

        grouped_cols = [c for c in ["flow_lt_lv", "flow_lt_se", "flow_lt_pl"] if c in filtered_df.columns]
        if grouped_cols:
            st.markdown("### Vidutiniai srautai pagal kryptį")
            flow_means = pd.DataFrame({
                "flow": grouped_cols,
                "mean": [filtered_df[c].mean() for c in grouped_cols]
            }).set_index("flow")
            st.bar_chart(flow_means)

        display_cols = [c for c in ["datetime"] + flow_cols if c in filtered_df.columns]
        st.dataframe(filtered_df[display_cols].tail(table_rows), use_container_width=True)

    with subtab4:
        st.markdown("### Heatmaps")
        st.caption("Heatmap braižoma tik iš filtruoto intervalo. Rekomenduojama rinktis trumpesnį laikotarpį.")

        heatmap_candidates = [
            c for c in [
                "price",
                "lv_price",
                "ee_price",
                "se4_price",
                "pl_price",
                "consumption_mw",
                "production_total_mw",
                "flow_total",
            ] if c in filtered_df.columns
        ]

        if not heatmap_candidates:
            st.info("Nėra tinkamų stulpelių heatmap grafikui.")
        else:
            heatmap_choice = st.selectbox(
                "Pasirink heatmap kintamąjį",
                options=heatmap_candidates
            )

            heatmap_df = filtered_df.copy()

            max_heatmap_days = 60
            unique_days = pd.Series(heatmap_df["date"]).nunique()

            if unique_days > max_heatmap_days:
                st.warning(
                    f"Pasirinktame intervale yra {unique_days} dienos. "
                    f"Heatmap ribojama iki paskutinių {max_heatmap_days} dienų."
                )
                last_dates = sorted(heatmap_df["date"].unique())[-max_heatmap_days:]
                heatmap_df = heatmap_df[heatmap_df["date"].isin(last_dates)].copy()

            make_heatmap(
                heatmap_df,
                heatmap_choice,
                f"{heatmap_choice} heatmap pagal datą ir valandą ({freq_choice})"
            )

    with subtab5:
        st.markdown("### Duomenų lentelė")
        st.dataframe(filtered_df.tail(table_rows), use_container_width=True)


# =========================
# PAGE: MODEL ANALYSIS
# =========================
def show_model_analysis(selected_version: str, metrics_df: pd.DataFrame):
    st.subheader("🤖 Model Analysis")

    if metrics_df.empty:
        st.warning("Nėra modelių metrikų šiai versijai.")
        return

    col1, col2 = st.columns(2)

    dataset_options = metrics_df["dataset_name"].drop_duplicates().tolist()
    model_options = metrics_df["model_name"].drop_duplicates().tolist()

    default_dataset = metrics_df.iloc[0]["dataset_name"]
    default_model = metrics_df.iloc[0]["model_name"]

    selected_dataset = col1.selectbox(
        "Pasirink dataset",
        options=dataset_options,
        index=dataset_options.index(default_dataset) if default_dataset in dataset_options else 0,
    )

    selected_model = col2.selectbox(
        "Pasirink modelį",
        options=model_options,
        index=model_options.index(default_model) if default_model in model_options else 0,
    )

    pred_df = get_predictions(selected_version, selected_dataset, selected_model)

    if pred_df.empty:
        st.warning("Nėra prognozių pasirinktam deriniui.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Eilučių", len(pred_df))
    c2.metric("MAE", f"{pred_df['abs_error'].mean():.2f}")
    c3.metric("Bias", f"{pred_df['error'].mean():.2f}")
    c4.metric("Max abs error", f"{pred_df['abs_error'].max():.2f}")

    st.markdown("### Faktinė vs prognozuota kaina")
    st.line_chart(
        pred_df.set_index("datetime")[["actual_price", "predicted_price"]],
        height=420
    )

    st.markdown("### Absoliuti klaida")
    st.line_chart(
        pred_df.set_index("datetime")[["abs_error"]],
        height=260
    )

    st.markdown("### Didžiausios klaidos")
    top_errors = pred_df.sort_values("abs_error", ascending=False).head(20).reset_index(drop=True)
    st.dataframe(top_errors, use_container_width=True)

    st.markdown("### Paskutinės prognozių eilutės")
    st.dataframe(pred_df.tail(30), use_container_width=True)


# =========================
# PAGE: DATA VERSIONS
# =========================
def show_data_versions(versions_df: pd.DataFrame, metrics_df: pd.DataFrame):
    st.subheader("🗄️ Data Versions")
    st.markdown("### Registruotos duomenų versijos")
    st.dataframe(versions_df, use_container_width=True)

    st.markdown("### Einamos versijos modelių metrikos")
    st.dataframe(metrics_df, use_container_width=True)


# =========================
# MAIN
# =========================
def main():
    st.title("⚡ Lietuvos elektros kainų prognozavimo sistema")

    versions_df = get_data_versions()

    if versions_df.empty:
        st.error("Duomenų bazėje nerasta nė vienos data_version. Pirma paleisk load_to_db.")
        return

    version_options = versions_df["version_name"].tolist()
    selected_version = st.sidebar.selectbox(
        "Pasirink data version",
        options=version_options,
        index=len(version_options) - 1,
    )

    selected_version_row = versions_df[versions_df["version_name"] == selected_version].iloc[0]

    st.sidebar.markdown("### Versijos informacija")
    st.sidebar.write(f"**Version:** {selected_version_row['version_name']}")
    st.sidebar.write(f"**Created:** {selected_version_row['created_at']}")
    st.sidebar.write(f"**Description:** {selected_version_row['description']}")

    page = st.sidebar.radio(
        "Navigacija",
        [
            "Dashboard",
            "Data Explorer",
            "Model Analysis",
            "Data Versions",
        ]
    )

    metrics_df = get_model_metrics(selected_version)

    if page == "Dashboard":
        show_dashboard(selected_version, metrics_df)
    elif page == "Data Explorer":
        show_data_explorer(selected_version)
    elif page == "Model Analysis":
        show_model_analysis(selected_version, metrics_df)
    elif page == "Data Versions":
        show_data_versions(versions_df, metrics_df)


if __name__ == "__main__":
    main()