# LT_elektros_kain-_prognozuotojas
streamlit run app/streamlit_app.py

Sukurti naują DB versiją:

python3 -m app.load_to_db --version-name Forecast_20260301_20260315 --description "Visi modeliai su prognoze nuo 2026-03-01 iki 2026-03-15"

python3 models/run_all_models.py \
  --train-start "2022-01-01" \
  --train-end "2026-02-28 23:59:59" \
  --test-start "2026-03-01" \
  --test-end "2026-03-15 23:59:59"

DB check paleidimas

  python3 -m app.db_check

 
 
  📌 Projekto aprašymas

Šio projekto tikslas – sukurti elektros energijos kainų prognozavimo sistemą, naudojant istorinius rinkos duomenis, meteorologinius duomenis ir mašininio mokymosi modelius.

Sistema apima:

duomenų surinkimą ir paruošimą,
feature engineering pipeline,
kelių modelių treniravimą,
rezultatų saugojimą duomenų bazėje,
interaktyvią analizę per Streamlit aplikaciją.
🏗️ Projekto struktūra
app/        → aplikacija, DB, Streamlit, helperiai  
data/       → duomenys (raw, processed, database)  
models/     → modelių treniravimo skriptai ir išsaugoti modeliai  
src/        → duomenų paruošimo pipeline (ETL + feature engineering)  
reports/    → modelių ir analizės rezultatai  
📂 Svarbiausi katalogai
app/
db.py – DB prisijungimas
db_check.py – DB tikrinimo skriptas
load_to_db.py – duomenų įkėlimas į DB
streamlit_app.py – vizualizacijos dashboard
training_utils.py – bendros treniravimo funkcijos
src/

Duomenų paruošimas:

prepare_master_data.py – pagrindinių kainų dataset
merge_nordpool_to_master.py – Nord Pool sujungimas
prepare_litgrid_data.py – vartojimas + gamyba
prepare_flows_data.py – tarpvalstybiniai srautai
prepare_weather_data.py – orų duomenys
features.py – feature engineering pipeline
models/

Modelių treniravimas:

train_xgb.py
train_lgbm.py
train_catboost.py
train_mlp.py
train_lstm.py
train_ensemble.py
run_all_models.py – paleidžia visus modelius
data/
raw/ – žali duomenys
processed/ – paruošti datasetai
database/ – SQLite DB
⚙️ Projekto paleidimas
1. Aplinka
pip install -r requirements.txt
2. Feature generavimas
python3 -m src.features
3. Modelių treniravimas
Paleisti visus modelius:
python3 -m models.run_all_models
Paleisti konkretų modelį:
python3 -m models.train_xgb
python3 -m models.train_lgbm
python3 -m models.train_catboost
python3 -m models.train_mlp
python3 -m models.train_lstm
4. Duomenų įkėlimas į DB
python3 -m app.load_to_db
5. DB patikra
python3 -m app.db_check
6. Streamlit aplikacija
streamlit run app/streamlit_app.py
🧠 Pipeline logika
Raw duomenys → src/
Sukuriamas master dataset
Generuojami features
Treniruojami modeliai (models/)
Rezultatai saugomi:
CSV
SQLite DB
Vizualizacija per Streamlit
📊 Naudojami modeliai
XGBoost
LightGBM
CatBoost
MLP
LSTM
Ensemble
🧪 Duomenys

Naudojami duomenys:

Elektros kainos (Litgrid / Nord Pool)
Vartojimas ir gamyba
Tarpvalstybiniai srautai
Meteorologiniai duomenys
⚠️ Svarbios pastabos
Visus skriptus paleisti su -m:
python3 -m app.db_check
Paleidimas iš projekto root katalogo
🚀 Greitas startas
pip install -r requirements.txt
python3 -m src.features
python3 -m models.run_all_models
python3 -m app.load_to_db
streamlit run app/streamlit_app.py
📈 Rezultatai

Modelių rezultatai saugomi:

data/processed/
reports/
SQLite DB
👨‍💻 Autorius

Šarūnas Grumadas