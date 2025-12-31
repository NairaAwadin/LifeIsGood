import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import pipeline_v as pv
import PE_dataviz2 as viz

DATA_PATH = "ressources/out_data/bienici/bienIci_listings.csv"

TARGET = pv.return_TARGET()
FEATURES = pv.return_FEATURES()
RENAME_MAP = pv.return_RENAME_MAP()
CAT_COLS = pv.return_CAT()
NUM_COLS = pv.return_NUM()


@st.cache_data
def load_raw_df(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


def build_analysis_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    X, y = pv.prepareDataset(df=raw_df)
    if y.name is None:
        y.name = pv.return_RENAME_MAP().get(pv.return_TARGET(), pv.return_TARGET())
    out = pd.concat([X, y], axis=1)
    return out


st.set_page_config(page_title="Bienetre - Analyse", layout="wide")
st.title("Bienetre - Analyse")

raw_df = load_raw_df(DATA_PATH)
if raw_df is None or raw_df.empty:
    st.warning("Aucun fichier ou aucune donnee trouvee dans ressources/out_data/bienici/bienIci_listings.csv.")
    st.stop()

analysis_df = build_analysis_df(raw_df)
if analysis_df.empty:
    st.warning("Aucune colonne exploitable apres selection features + target.")
    st.stop()

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.subheader("Distribution du prix au m2")
    postal_col = "postal_code"
    if postal_col in analysis_df.columns:
        postal_codes = sorted(analysis_df[postal_col].dropna().astype(str).unique().tolist())
        if postal_codes:
            postal_choice = st.selectbox("Code postal", options=["Tous"] + postal_codes, index=0)
        else:
            postal_choice = "Tous"
    else:
        postal_choice = "Tous"

    try:
        pc_filter = None if postal_choice == "Tous" else postal_choice
        fig, _, _, _ = viz.plot_distribution_prix_m2_appartements(
            analysis_df,
            postalCode=pc_filter,
            postal_col="postal_code",
            col_type="property_type",
            col_prix_m2="price_per_sqm",
        )
        st.pyplot(fig, clear_figure=True)
    except Exception as exc:
        st.error(str(exc))

with col_right:
    st.subheader("Regression lineaire")
    scatter_options = [
        "computedSurfaceArea",
        "roomsQuantity",
        "computedBedroomsQuantity",
        "floor",
        "computedBathroomsQuantity",
        "computedShowerRoomsQuantity",
        "computedToiletQuantity",
        "computedEnergyValue",
        "computedMinEnergyConsumption",
        "computedMaxEnergyConsumption",
        "greenhouseGazValue",
    ]
    renamed_options = [RENAME_MAP.get(c, c) for c in scatter_options]
    renamed_options = [c for c in renamed_options if c in analysis_df.columns]

    if not renamed_options:
        st.warning("Aucune colonne numerique disponible pour la regression.")
    else:
        x_choice = st.selectbox("Variable X", options=renamed_options, index=0)
        try:
            fig, _, _, _, _ = viz.plot_scatter_regression(analysis_df, x_col=x_choice, y_col="price_per_sqm")
            st.pyplot(fig, clear_figure=True)
        except Exception as exc:
            st.error(str(exc))

st.subheader("Carte prix au m2 et densite")
try:
    map_obj = viz.carte_prix_couleur_densite_taille(
        analysis_df,
        postal_col="postal_code",
        lat_col="lat",
        lon_col="lon",
        price_m2_col="price_per_sqm",
    )
    components.html(map_obj._repr_html_(), height=700, scrolling=True)
except Exception as exc:
    st.error(str(exc))
