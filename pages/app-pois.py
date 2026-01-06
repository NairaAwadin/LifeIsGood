import osm
from pathlib import Path

import pandas as pd
import streamlit as st


def get_latest_ds(ds_root: Path) -> Path | None:
    if not ds_root.exists():
        return None
    candidates = []
    for f in ds_root.iterdir():
        if f.is_file() and f.suffix.lower() == ".csv":
            candidates.append((f.stat().st_mtime, f))
    if not candidates:
        for f in ds_root.rglob("*.csv"):
            if f.is_file():
                candidates.append((f.stat().st_mtime, f))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def get_data(data_path: str = "ressources/out_data/bienici") -> pd.DataFrame:
    path = get_latest_ds(ds_root=Path(data_path))
    if path is None:
        return pd.DataFrame()
    return pd.read_csv(path)


if "poi_payload" not in st.session_state:
    st.session_state.poi_payload = None
if "poi_coords" not in st.session_state:
    st.session_state.poi_coords = None


st.set_page_config(page_title="Bienetre - POI", layout="wide")
st.title("Bienetre - POI")
st.subheader("Recherche POI")
st.caption("Chercher les POI proches a partir d'un id d'annonce du snapshot.")

with st.form("pois_form"):
    search_id = st.text_input("id annonce")
    pois_submitted = st.form_submit_button("Analyser l'approximation")

if pois_submitted:
    df = get_data()
    if df is None or df.empty:
        st.warning("Aucune donnee disponible pour la recherche.")
    else:
        mask = df["id"].astype(str) == str(search_id)
        if not mask.any():
            st.warning("Aucune annonce trouvee pour cet id.")
        else:
            row = df.loc[mask].iloc[0]
            lat = row.get("blurInfo.centroid.lat")
            lon = row.get("blurInfo.centroid.lon")

            if pd.isna(lat) or pd.isna(lon):
                st.warning("Latitude/longitude manquantes pour cette annonce.")
            else:
                try:
                    pois_dict = osm.get_pois(lat=float(lat), lon=float(lon))
                except Exception as exc:
                    st.error(f"Erreur OSM : {exc}")
                else:
                    st.session_state.poi_payload = pois_dict
                    st.session_state.poi_coords = (lat, lon)

if st.session_state.poi_payload is not None:
    lat, lon = st.session_state.poi_coords
    coord_cols = st.columns(2)
    coord_cols[0].metric("Latitude", f"{lat:.6f}" if pd.notna(lat) else "NA")
    coord_cols[1].metric("Longitude", f"{lon:.6f}" if pd.notna(lon) else "NA")

    pois_dict = st.session_state.poi_payload
    if not pois_dict:
        st.info("Aucun POI renvoye.")
    elif not isinstance(pois_dict, dict):
        st.write(pois_dict)
    else:
        def render_value(value):
            if isinstance(value, list):
                st.dataframe(pd.DataFrame(value), use_container_width=True)
            elif isinstance(value, dict):
                st.dataframe(pd.DataFrame([value]), use_container_width=True)
            else:
                st.write(value)

        other = {k: v for k, v in pois_dict.items() if k not in ("poi_categories", "nearest")}
        if other:
            with st.expander("Infos generales", expanded=True):
                for k, v in other.items():
                    st.markdown(f"**{k}**")
                    render_value(v)

        nearest = pois_dict.get("nearest", [])
        if nearest:
            st.subheader("POI les plus proches")
            if isinstance(nearest, dict):
                for k, v in nearest.items():
                    st.markdown(f"**{k}**")
                    render_value(v)
            elif isinstance(nearest, list):
                for i, item in enumerate(nearest, start=1):
                    st.markdown(f"Proche {i}")
                    if isinstance(item, dict):
                        st.dataframe(pd.DataFrame([item]), use_container_width=True)
                    else:
                        st.write(item)
            else:
                render_value(nearest)

        poi_categories = pois_dict.get("poi_categories", {})
        if poi_categories:
            st.subheader("Categories de POI")
            for k, v in poi_categories.items():
                count = v.get("count", 0)
                nearest = v.get("nearest", [])
                with st.expander(f"{k} (count: {count})", expanded=False):
                    render_value(nearest)
        else:
            st.info("Aucune categorie de POI renvoyee.")
