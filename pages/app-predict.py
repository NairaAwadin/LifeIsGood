import streamlit as st
import pandas as pd
import numpy as np

import get_catboost as cat
import pipeline_v as pv

MODEL_PATH = "models/catboost/catboost_v1.cbm"

RENAME_MAP = pv.return_RENAME_MAP()
CAT_COLS = pv.return_CAT()
NUM_COLS = pv.return_NUM()
BOOL_COLS = pv.return_BOOL()
FEATURES = pv.return_FEATURES()
TARGET = pv.return_TARGET()

REQUIRED_NORM = [
    "transaction_type",
    "postal_code",
    "surface_area",
    "property_type",
    "rooms",
    "bedrooms",
]

# reverse mapping: normalized -> original
NORM_TO_ORIG = {v: k for k, v in RENAME_MAP.items()}

PROPERTY_TYPES = ["flat"]
CITY_CHOICES = ["paris", "lyon", "marseille"]
HEATING_OPTIONS = ["__UNKNOWN__", "collectif", "individuel", "gaz", "electrique", "fioul", "bois", "autre"]
EXPOSITION_OPTIONS = [
    "__UNKNOWN__",
    "nord",
    "sud",
    "est",
    "ouest",
    "nord-est",
    "nord-ouest",
    "sud-est",
    "sud-ouest",
]
ENERGY_CLASS_OPTIONS = ["__UNKNOWN__", "A", "B", "C", "D", "E", "F", "G"]
GHG_CLASS_OPTIONS = ["__UNKNOWN__", "A", "B", "C", "D", "E", "F", "G"]

POSTAL_CODE_OPTIONS = {
    "paris": [str(x) for x in range(75001, 75021)],
    "lyon": [str(x) for x in range(69001, 69010)],
    "marseille": [str(x) for x in range(13001, 13017)],
}


@st.cache_resource
def load_model():
    model = cat.load_catboost(MODEL_PATH)
    feature_names = model.feature_names_
    return model, feature_names


def build_input_row(form_data: dict) -> pd.DataFrame:
    row = {}
    for norm_key, value in form_data.items():
        orig_key = NORM_TO_ORIG.get(norm_key, norm_key)
        row[orig_key] = value
    return pd.DataFrame([row])


def prepare_for_model(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    X = pv.prepareX(df)
    X = X.rename(columns=RENAME_MAP)
    # ensure model feature order
    for col in feature_names:
        if col not in X.columns:
            X[col] = pd.NA
    X = X[feature_names]
    return X


st.set_page_config(page_title="Bienetre - Prediction", layout="wide")
st.title("Bienetre - Prediction")

model, feature_names = load_model()

st.caption("Entrez les caracteristiques du bien pour estimer le loyer.")

with st.form("predict_form"):
    st.subheader("Champs requis")
    left, right = st.columns(2)

    with left:
        transaction_type = st.selectbox("Type de transaction", options=["rent"], index=0)
        city = st.selectbox("Ville", options=CITY_CHOICES, index=0)
        postal_options = POSTAL_CODE_OPTIONS.get(city, [])
        postal_code = st.selectbox("Code postal", options=postal_options, index=0)
        property_type = st.selectbox("Type de bien", options=PROPERTY_TYPES, index=0)
        surface_area = st.number_input("Surface (m2)", min_value=5.0, value=30.0, step=1.0)
        rooms = st.number_input("Pieces", min_value=0.0, value=1.0, step=1.0)

    with right:
        bedrooms = st.number_input("Chambres", min_value=0.0, value=1.0, step=1.0)

    with st.expander("Avance (optionnel)", expanded=False):
        adv_left, adv_right = st.columns(2)

        with adv_left:
            lat = st.number_input("Latitude", value=48.8566, format="%.6f")
            lon = st.number_input("Longitude", value=2.3522, format="%.6f")
            floor = st.number_input("Etage", value=0.0, step=1.0)
            is_new = st.checkbox("Neuf", value=False)
            has_elevator = st.checkbox("Ascenseur", value=False)
            has_terrace = st.checkbox("Terrasse", value=False)
            heating = st.selectbox("Chauffage", options=HEATING_OPTIONS, index=0)
            exposition = st.selectbox("Exposition", options=EXPOSITION_OPTIONS, index=0)

        with adv_right:
            bathrooms = st.number_input("Salles de bain", value=0.0, step=1.0)
            shower_rooms = st.number_input("Salles d'eau", value=0.0, step=1.0)
            toilets = st.number_input("Toilettes", value=1.0, step=1.0)
            energy_class = st.selectbox("Classe energie", options=ENERGY_CLASS_OPTIONS, index=0)
            energy_value = st.number_input("Energie (kWh/m2/an)", value=0.0, step=1.0)
            energy_min = st.number_input("Energie min", value=0.0, step=1.0)
            energy_max = st.number_input("Energie max", value=0.0, step=1.0)
            ghg_class = st.selectbox("Classe GES", options=GHG_CLASS_OPTIONS, index=0)
            ghg_value = st.number_input("GES", value=0.0, step=1.0)

    submitted = st.form_submit_button("Predire")

if submitted:
    form_data = {
        "transaction_type": transaction_type,
        "city": city,
        "postal_code": postal_code,
        "property_type": property_type,
        "surface_area": surface_area,
        "rooms": rooms,
        "bedrooms": bedrooms,
        "lat": lat,
        "lon": lon,
        "floor": floor,
        "is_new": is_new,
        "has_elevator": has_elevator,
        "has_terrace": has_terrace,
        "heating": heating,
        "exposition": exposition,
        "bathrooms": bathrooms,
        "shower_rooms": shower_rooms,
        "toilets": toilets,
        "energy_class": energy_class,
        "energy_value": energy_value,
        "energy_min": energy_min,
        "energy_max": energy_max,
        "ghg_class": ghg_class,
        "ghg_value": ghg_value,
    }

    raw_df = build_input_row(form_data)
    X = prepare_for_model(raw_df, feature_names)

    cat_cols_norm = [RENAME_MAP.get(c, c) for c in CAT_COLS]
    pred = cat.predict(model=model, X=X, cat_cols=cat_cols_norm)

    price_per_sqm = float(pred[0])
    monthly_rent = price_per_sqm * float(surface_area)

    st.subheader("Prediction")
    cols = st.columns(2)
    cols[0].metric("Loyer mensuel estime", f"{monthly_rent:,.2f}")
    cols[1].metric("Loyer mensuel au m2", f"{price_per_sqm:,.2f}")

    with st.expander("Entree preparee", expanded=False):
        st.dataframe(X, use_container_width=True)
