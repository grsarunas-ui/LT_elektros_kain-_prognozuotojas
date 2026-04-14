import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# ĮVESTIES FAILAI
# =========================================================
# Pagrindiniai jau apdoroti duomenų failai su Lietuvos kaina + Nord Pool kainomis
INPUT_15MIN = Path("data/processed/master_with_nordpool_15min.csv")
INPUT_HOURLY = Path("data/processed/master_with_nordpool_hourly.csv")

# =========================================================
# IŠVESTIES FAILAI
# =========================================================
# Clean variantas – be papildomų orų ir išplėstų Nord Pool požymių
OUTPUT_15MIN_CLEAN = Path("data/processed/features_15min_clean.csv")
OUTPUT_15MIN_EXTENDED = Path("data/processed/features_15min_extended.csv")

OUTPUT_HOURLY_CLEAN = Path("data/processed/features_hourly_clean.csv")
OUTPUT_HOURLY_EXTENDED = Path("data/processed/features_hourly_extended.csv")

# =========================================================
# PAPILDOMI DUOMENŲ ŠALTINIAI
# =========================================================
# Litgrid vartojimo ir gamybos duomenys
LITGRID_15MIN = Path("data/processed/litgrid_features_15min.csv")
LITGRID_HOURLY = Path("data/processed/litgrid_features_hourly.csv")

# Tarpvalstybiniai srautai
FLOWS_15MIN = Path("data/processed/flows_15min.csv")
FLOWS_HOURLY = Path("data/processed/flows_hourly.csv")

# Oro duomenys (naudojami tik extended režime)
WEATHER_HOURLY = Path("data/processed/weather_lithuania_avg_hourly.csv")


def detect_frequency(df: pd.DataFrame) -> str:
    """
    Automatiškai nustato duomenų dažnį pagal datetime skirtumus.
    
    Jei dažniausias žingsnis yra <= 15 min, laikome, kad tai 15 min duomenys.
    Kitu atveju – valandiniai duomenys.
    """
    diffs = df["datetime"].diff().dropna()

    if diffs.empty:
        # Jei duomenų labai mažai, pagal nutylėjimą laikome hourly
        return "hourly"

    mode_diff = diffs.mode().iloc[0]

    if mode_diff <= pd.Timedelta(minutes=15):
        return "15min"

    return "hourly"


def get_time_config(freq: str) -> dict:
    """
    Grąžina konfigūraciją, priklausomai nuo duomenų dažnio.
    
    Čia apibrėžiami:
    - kokie lag'ai naudojami,
    - kiek žingsnių atitinka 24h, 48h ir 7 dienas,
    - kokio ilgio rolling langai.
    """
    if freq == "15min":
        return {
            # Lag'ai 15 min duomenims:
            # 1 = 15 min atgal
            # 4 = 1 val. atgal
            # 8 = 2 val. atgal
            # 96 = 24 val. atgal
            # 192 = 48 val. atgal
            # 672 = 7 dienos atgal
            "price_lags": [1, 4, 8, 96, 192, 672],

            # Alternatyvus testuotas variantas tik su ilgesniais lag'ais:
            # "price_lags": [96, 192, 672],

            "lag_24h": 96,
            "lag_48h": 192,
            "lag_7d": 672,
            "roll_24h": 96,
            "roll_7d": 672,
            "short_shift": 3,
            "medium_shift": 6,
        }
    else:
        return {
            # Lag'ai valandiniams duomenims:
            # 1 = 1 val. atgal
            # 24 = 24 val. atgal
            # 168 = 7 dienos atgal
            "price_lags": [1, 2, 3, 24, 48, 72, 168],

            # Alternatyvus testuotas variantas tik su ilgesniais lag'ais:
            # "price_lags": [24, 48, 72, 168],

            "lag_24h": 24,
            "lag_48h": 48,
            "lag_7d": 168,
            "roll_24h": 24,
            "roll_7d": 168,
            "short_shift": 3,
            "medium_shift": 6,
        }


def add_time_features(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Sukuria laiko požymius iš datetime stulpelio.
    
    Tikslas:
    - modeliuoti dienos, savaitės ir metų sezoniškumą,
    - padėti modeliui suprasti, kada yra pikinės valandos ar savaitgaliai.
    """
    df = df.copy()

    # Baziniai kalendoriniai požymiai
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["weekday"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month
    df["dayofyear"] = df["datetime"].dt.dayofyear
    df["weekofyear"] = df["datetime"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # Valandos pozicija savaitėje
    df["hour_week"] = df["weekday"] * 24 + df["hour"]

    # Rankiniu būdu pažymimos tipinės didesnio vartojimo valandos
    df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    # Ciklinės transformacijos – kad modelis suprastų, jog pvz. 23 val. arti 0 val.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    df["dayofyear_sin"] = np.sin(2 * np.pi * (df["dayofyear"] - 1) / 365.25)
    df["dayofyear_cos"] = np.cos(2 * np.pi * (df["dayofyear"] - 1) / 365.25)

    # 15 min duomenims pridedamas tikslesnis dienos ciklo požymis
    if freq == "15min":
        quarter = (df["hour"] * 60 + df["minute"]) / (24 * 60)
        df["timeofday_sin"] = np.sin(2 * np.pi * quarter)
        df["timeofday_cos"] = np.cos(2 * np.pi * quarter)

    return df


def add_price_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Sukuria kainos istorijos ir volatilumo požymius.
    
    Tai viena svarbiausių feature grupių, nes elektros kaina labai priklauso
    nuo savo ankstesnių reikšmių, trendų ir nesenų svyravimų.
    """
    df = df.copy()

    # Sukuriami lag'ai pagal pasirinktą konfigūraciją
    for lag in cfg["price_lags"]:
        df[f"lag_{lag}"] = df["price"].shift(lag)

    # Lag 1 ir lag 2 naudojami trumpalaikių pokyčių skaičiavimui
    lag_1 = df["lag_1"] if "lag_1" in df.columns else df["price"].shift(1)
    lag_2 = df["lag_2"] if "lag_2" in df.columns else df["price"].shift(2)

    # Vieno žingsnio kainos pokytis
    df["price_diff_1"] = lag_1 - lag_2

    # Pokytis lyginant su kaina prieš 24h
    if f"lag_{cfg['lag_24h']}" in df.columns:
        df["price_diff_24h"] = lag_1 - df[f"lag_{cfg['lag_24h']}"]
        df["trend_24h"] = lag_1 - df[f"lag_{cfg['lag_24h']}"]

    # Pokytis lyginant su prieš savaitę buvusia kaina
    if f"lag_{cfg['lag_7d']}" in df.columns:
        df["trend_7d"] = lag_1 - df[f"lag_{cfg['lag_7d']}"]

    # Slenkantys vidurkiai – rodo „normalų“ kainos lygį
    df["rolling_mean_24h"] = df["price"].shift(1).rolling(cfg["roll_24h"]).mean()
    df["rolling_mean_7d"] = df["price"].shift(1).rolling(cfg["roll_7d"]).mean()

    # Slenkantys standartiniai nuokrypiai – rodo volatilumą
    df["rolling_std_24h"] = df["price"].shift(1).rolling(cfg["roll_24h"]).std()
    df["rolling_std_7d"] = df["price"].shift(1).rolling(cfg["roll_7d"]).std()

    # Min/max per 24h – naudinga nustatyti kainos padėtį intervalo ribose
    df["rolling_min_24h"] = df["price"].shift(1).rolling(cfg["roll_24h"]).min()
    df["rolling_max_24h"] = df["price"].shift(1).rolling(cfg["roll_24h"]).max()

    # Kainos santykis su savo nesenu vidurkiu
    df["price_vs_mean24h"] = lag_1 / (df["rolling_mean_24h"] + 1e-6)
    df["price_vs_mean7d"] = lag_1 / (df["rolling_mean_7d"] + 1e-6)

    # Ar tai galimas spike'as?
    df["price_spike_flag"] = (lag_1 > df["rolling_mean_24h"] * 1.5).astype(int)

    # Trumpalaikės ir vidutinės trukmės tendencijos
    df["trend_short"] = lag_1 - df["price"].shift(cfg["short_shift"])
    df["trend_medium"] = lag_1 - df["price"].shift(cfg["medium_shift"])

    # Standartizuotas nukrypimas nuo 24h vidurkio
    df["price_zscore_24h"] = (
        (lag_1 - df["rolling_mean_24h"]) /
        (df["rolling_std_24h"] + 1e-6)
    )

    # Kainos pozicija 24h intervalo ribose (nuo 0 iki 1)
    df["range_position_24h"] = (
        (lag_1 - df["rolling_min_24h"]) /
        (df["rolling_max_24h"] - df["rolling_min_24h"] + 1e-6)
    )

    # Tiesiogiai išsaugomas volatilumo lygis
    df["volatility_24h"] = df["rolling_std_24h"]
    df["volatility_7d"] = df["rolling_std_7d"]

    # Staigus kainos pokytis per vieną žingsnį
    df["price_jump_1"] = lag_1 - lag_2
    df["abs_price_jump_1"] = df["price_jump_1"].abs()

    # Santykis su vakarykšte kaina
    if f"lag_{cfg['lag_24h']}" in df.columns:
        df["lag1_vs_lag24"] = lag_1 - df[f"lag_{cfg['lag_24h']}"]
        df["price_return_24h"] = lag_1 / (df[f"lag_{cfg['lag_24h']}"] + 1e-6)

    # Santykis su ankstesniu žingsniu
    df["price_return_1"] = lag_1 / (lag_2 + 1e-6)

    # Ar dabar rinkoje didesnis volatilumas nei įprastai?
    df["high_vol_regime"] = (
        df["volatility_24h"] >
        df["volatility_24h"].shift(1).rolling(cfg["roll_7d"]).mean()
    ).astype(int)

    return df


def add_nordpool_features(
    df: pd.DataFrame,
    cfg: dict,
    include_spreads: bool = True,
    keep_raw_prices: bool = False
) -> pd.DataFrame:
    """
    Sukuria papildomus požymius iš kaimyninių Nord Pool zonų kainų.
    
    Tikslas:
    - įtraukti regioninį kainų kontekstą,
    - modeliuoti skirtumus tarp Lietuvos ir kaimyninių rinkų.
    """
    df = df.copy()

    nordpool_cols = ["lv_price", "ee_price", "se4_price", "pl_price"]

    for col in nordpool_cols:
        if col not in df.columns:
            continue

        df[col] = pd.to_numeric(df[col], errors="coerce")

        # Kaimyninių rinkų lag'ai
        df[f"{col}_lag_24h"] = df[col].shift(cfg["lag_24h"])
        df[f"{col}_lag_48h"] = df[col].shift(cfg["lag_48h"])
        df[f"{col}_lag_7d"] = df[col].shift(cfg["lag_7d"])

        # Kaimyninių rinkų rolling statistika
        df[f"{col}_rolling_mean_24h"] = df[col].shift(1).rolling(cfg["roll_24h"]).mean()
        df[f"{col}_rolling_std_24h"] = df[col].shift(1).rolling(cfg["roll_24h"]).std()

        # Pokytis lyginant su vakarykšte tos rinkos kaina
        df[f"{col}_diff_24h"] = df[col].shift(1) - df[col].shift(cfg["lag_24h"])

    if include_spreads:
        lt_lag_1 = df["price"].shift(1)

        # Kainų skirtumai tarp Lietuvos ir kitų rinkų
        if "lv_price" in df.columns:
            df["spread_lv"] = lt_lag_1 - df["lv_price"].shift(1)
            df["spread_lv_change"] = df["spread_lv"] - df["spread_lv"].shift(1)
            df["spread_lv_zscore"] = (
                (df["spread_lv"] - df["spread_lv"].shift(1).rolling(cfg["roll_24h"]).mean()) /
                (df["spread_lv"].shift(1).rolling(cfg["roll_24h"]).std() + 1e-6)
            )

        if "ee_price" in df.columns:
            df["spread_ee"] = lt_lag_1 - df["ee_price"].shift(1)

        if "se4_price" in df.columns:
            df["spread_se4"] = lt_lag_1 - df["se4_price"].shift(1)

        if "pl_price" in df.columns:
            df["spread_pl"] = lt_lag_1 - df["pl_price"].shift(1)

    # Jei nenorime laikyti žalių Nord Pool kainų, jas pašaliname
    if not keep_raw_prices:
        drop_cols = [c for c in nordpool_cols if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

    return df


def add_litgrid_features(df: pd.DataFrame, freq: str, cfg: dict) -> pd.DataFrame:
    """
    Prijungia Litgrid vartojimo ir gamybos duomenis bei sukuria jų išvestinius požymius.
    
    Tikslas:
    - įtraukti elektros sistemos balansą,
    - apskaičiuoti net load ir jo pokyčius.
    """
    df = df.copy()

    litgrid_path = LITGRID_15MIN if freq == "15min" else LITGRID_HOURLY

    if not litgrid_path.exists():
        print(f"⚠️ Litgrid failas nerastas: {litgrid_path}")
        return df

    lit = pd.read_csv(litgrid_path)
    lit["datetime"] = pd.to_datetime(lit["datetime"], errors="coerce")
    lit = lit.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    # Sujungiame su pagrindiniu rinkiniu pagal datetime
    df = df.merge(lit, on="datetime", how="left")

    # Grynasis apkrovimas = vartojimas - gamyba
    if "consumption_mw" in df.columns and "production_total_mw" in df.columns:
        df["net_load_mw"] = df["consumption_mw"] - df["production_total_mw"]

    base_cols = [
        "consumption_mw",
        "production_total_mw",
        "net_load_mw",
    ]

    existing_base_cols = [c for c in base_cols if c in df.columns]

    # Kuriame lag'us ir rolling statistiką
    for col in existing_base_cols:
        df[f"{col}_lag_1"] = df[col].shift(1)
        df[f"{col}_lag_24h"] = df[col].shift(cfg["lag_24h"])
        df[f"{col}_lag_48h"] = df[col].shift(cfg["lag_48h"])
        df[f"{col}_lag_7d"] = df[col].shift(cfg["lag_7d"])
        df[f"{col}_rolling_mean_24h"] = df[col].shift(1).rolling(cfg["roll_24h"]).mean()
        df[f"{col}_rolling_std_24h"] = df[col].shift(1).rolling(cfg["roll_24h"]).std()

    # Pokyčiai per 24h
    if "consumption_mw" in df.columns:
        df["consumption_diff_24h"] = (
            df["consumption_mw"].shift(1) - df["consumption_mw"].shift(cfg["lag_24h"])
        )

    if "production_total_mw" in df.columns:
        df["production_diff_24h"] = (
            df["production_total_mw"].shift(1) - df["production_total_mw"].shift(cfg["lag_24h"])
        )

    if "net_load_mw" in df.columns:
        df["net_load_diff_24h"] = (
            df["net_load_mw"].shift(1) - df["net_load_mw"].shift(cfg["lag_24h"])
        )

    # Pašaliname pirminius stulpelius, paliekame tik išvestinius požymius
    df = df.drop(columns=existing_base_cols, errors="ignore")
    return df


def add_flows_features(df: pd.DataFrame, freq: str, cfg: dict) -> pd.DataFrame:
    """
    Prijungia tarpvalstybinių srautų duomenis ir sukuria jų išvestinius požymius.
    
    Tikslas:
    - modeliuoti importo / eksporto dinamiką,
    - įtraukti regioninio balanso poveikį kainai.
    """
    df = df.copy()

    flows_path = FLOWS_15MIN if freq == "15min" else FLOWS_HOURLY

    if not flows_path.exists():
        print(f"⚠️ Flows failas nerastas: {flows_path}")
        return df

    flows = pd.read_csv(flows_path)
    flows["datetime"] = pd.to_datetime(flows["datetime"], errors="coerce")
    flows = flows.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    df = df.merge(flows, on="datetime", how="left")

    base_cols = [
        "flow_lt_lv",
        "flow_lt_se",
        "flow_lt_pl",
        "flow_total",
        "flow_abs_total",
    ]

    existing_base_cols = [c for c in base_cols if c in df.columns]

    for col in existing_base_cols:
        df[f"{col}_lag_1"] = df[col].shift(1)
        df[f"{col}_lag_24h"] = df[col].shift(cfg["lag_24h"])
        df[f"{col}_lag_48h"] = df[col].shift(cfg["lag_48h"])
        df[f"{col}_lag_7d"] = df[col].shift(cfg["lag_7d"])
        df[f"{col}_rolling_mean_24h"] = df[col].shift(1).rolling(cfg["roll_24h"]).mean()
        df[f"{col}_rolling_std_24h"] = df[col].shift(1).rolling(cfg["roll_24h"]).std()

    # Bendrų srautų pokyčiai per parą
    if "flow_total" in df.columns:
        df["flow_total_diff_24h"] = (
            df["flow_total"].shift(1) - df["flow_total"].shift(cfg["lag_24h"])
        )

    if "flow_abs_total" in df.columns:
        df["flow_abs_total_diff_24h"] = (
            df["flow_abs_total"].shift(1) - df["flow_abs_total"].shift(cfg["lag_24h"])
        )

    # Paliekame tik išvestinius požymius
    df = df.drop(columns=existing_base_cols, errors="ignore")
    return df


def add_weather_features(df: pd.DataFrame, freq: str, cfg: dict) -> pd.DataFrame:
    """
    Prijungia orų duomenis ir sukuria jų išvestinius požymius.
    
    Naudojama tik extended režime.
    Tikslas:
    - įtraukti temperatūros, vėjo, debesuotumo ir saulės poveikį kainai,
    - papildomai modeliuoti atsinaujinančios energijos potencialą.
    """
    df = df.copy()

    if not WEATHER_HOURLY.exists():
        print(f"⚠️ Weather failas nerastas: {WEATHER_HOURLY}")
        return df

    weather = pd.read_csv(WEATHER_HOURLY)
    weather["datetime"] = pd.to_datetime(weather["datetime"], errors="coerce")
    weather = weather.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    weather_cols = [
        "temperature_2m",
        "wind_speed_10m",
        "cloud_cover",
        "shortwave_radiation",
        "solar_proxy",
    ]
    existing_weather_cols = [c for c in weather_cols if c in weather.columns]
    if not existing_weather_cols:
        return df

    weather = weather[["datetime"] + existing_weather_cols].copy()

    # Jei pagrindinis datasetas yra 15 min, hourly orų duomenys persamplinami į 15 min
    if freq == "15min":
        weather = (
            weather.set_index("datetime")
            .resample("15min")
            .ffill()
            .reset_index()
        )

    df = df.merge(weather, on="datetime", how="left")

    for col in existing_weather_cols:
        df[f"{col}_lag_1"] = df[col].shift(1)
        df[f"{col}_lag_24h"] = df[col].shift(cfg["lag_24h"])
        df[f"{col}_lag_48h"] = df[col].shift(cfg["lag_48h"])
        df[f"{col}_rolling_mean_24h"] = df[col].shift(1).rolling(cfg["roll_24h"]).mean()
        df[f"{col}_rolling_std_24h"] = df[col].shift(1).rolling(cfg["roll_24h"]).std()

    # Kombinuotas vėjo ir saulės poveikio požymis
    if "wind_speed_10m" in df.columns and "shortwave_radiation" in df.columns:
        df["wind_solar_interaction"] = (
            df["wind_speed_10m"].shift(1) * df["shortwave_radiation"].shift(1)
        )

    # Saulės potencialas, pakoreguotas pagal debesuotumą
    if "cloud_cover" in df.columns and "shortwave_radiation" in df.columns:
        df["solar_cloud_adjusted"] = (
            df["shortwave_radiation"].shift(1) *
            (1 - df["cloud_cover"].shift(1) / 100.0)
        )

    # Pašalinami pirminiai orų stulpeliai
    df = df.drop(columns=existing_weather_cols, errors="ignore")
    return df


def create_features(df: pd.DataFrame, mode: str = "clean") -> pd.DataFrame:
    """
    Pagrindinė feature engineering funkcija.
    
    Režimai:
    - clean: baziniai feature (time + price + litgrid + flows)
    - extended: papildomai įtraukiami weather ir Nord Pool išvestiniai feature
    """
    if mode not in {"clean", "extended"}:
        raise ValueError("mode turi būti 'clean' arba 'extended'")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    # Nustatome, ar dirbame su 15 min ar hourly duomenimis
    freq = detect_frequency(df)
    cfg = get_time_config(freq)

    use_weather = mode == "extended"
    use_nordpool_extended = mode == "extended"

    print(f"Detected frequency: {freq}")
    print(f"Mode: {mode}")
    print(f"Use weather: {use_weather}")
    print(f"Use nordpool extended: {use_nordpool_extended}")
    print(f"Price lags: {cfg['price_lags']}")

    # Požymiai kuriami etapais
    df = add_time_features(df, freq)
    df = add_price_features(df, cfg)
    df = add_litgrid_features(df, freq, cfg)
    df = add_flows_features(df, freq, cfg)

    if use_weather:
        df = add_weather_features(df, freq, cfg)

    if use_nordpool_extended:
        df = add_nordpool_features(df, cfg, include_spreads=True)

    # Sutvarkome netinkamas reikšmes
    df = df.replace([np.inf, -np.inf], np.nan)

    # Pašaliname eilutes, kuriose dar liko NaN
    # Dažniausiai jos atsiranda dėl lag / rolling skaičiavimų pradžioje
    df = df.dropna().reset_index(drop=True)

    return df


def process_file(input_path: Path, output_path: Path, mode: str) -> None:
    """
    Apdoroja vieną įėjimo failą:
    1. nuskaito duomenis,
    2. sukuria feature,
    3. išsaugo rezultatą į CSV.
    """
    if not input_path.exists():
        print(f"Nerastas failas: {input_path}")
        return

    df = pd.read_csv(input_path)

    print("\n" + "=" * 80)
    print(f"Apdorojamas: {input_path}")
    print(f"Režimas: {mode}")
    print("Pradinė forma:", df.shape)
    print("Stulpeliai:", df.columns.tolist())

    df_features = create_features(df, mode=mode)

    print("Po feature engineering:", df_features.shape)
    print("Laikotarpis:", df_features["datetime"].min(), "->", df_features["datetime"].max())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False)

    print(f"✓ Išsaugota: {output_path}")


if __name__ == "__main__":
    # Sugeneruojami visi 4 feature failai:
    # 15 min clean
    # 15 min extended
    # hourly clean
    # hourly extended
    process_file(INPUT_15MIN, OUTPUT_15MIN_CLEAN, mode="clean")
    process_file(INPUT_15MIN, OUTPUT_15MIN_EXTENDED, mode="extended")
    process_file(INPUT_HOURLY, OUTPUT_HOURLY_CLEAN, mode="clean")
    process_file(INPUT_HOURLY, OUTPUT_HOURLY_EXTENDED, mode="extended")