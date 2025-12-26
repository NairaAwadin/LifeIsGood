import get_catboost as cat
import get_DistilCamemBERT as camembert
import osm
import pipeline_0 as pl0
import pipeline_v as plv
import scrape_bienici as scrapeBi
from pathlib import Path
import os

import pandas as pd
import numpy as np
import streamlit as st


displayed_features = [
    "id",
    "advertisedPrice",
    "pricePerSquareMeter",
    "city",
    "computedPostalCode",
    "surfaceArea",
    "roomsQuantity",
    "bedroomsQuantity",
    "energyClassification",
    "greenhouseGazClassification",
    "description",
    "falseLocation",
]

tokenizer, model = camembert.load_model("models/location_flagger_v4")
camembert_bundle = {"model": model, "tokenizer": tokenizer}


def get_latest_ds(ds_root: Path) -> Path | None:
    if not ds_root.exists():
        return None
    candidates = []
    for f in ds_root.iterdir():
        if f.is_file() and f.suffix.lower() == ".csv":
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


if "scrape_results" not in st.session_state:
    st.session_state.scrape_results = None
if "scrape_status" not in st.session_state:
    st.session_state.scrape_status = ""
if "scrape_progress" not in st.session_state:
    st.session_state.scrape_progress = 0.0
if "poi_payload" not in st.session_state:
    st.session_state.poi_payload = None
if "poi_coords" not in st.session_state:
    st.session_state.poi_coords = None


st.set_page_config(page_title="Bienetre", layout="wide")
st.title("Bienetre")

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("Scraper les annonces")
    st.caption("Recuperer les pages Bienici, executer le pipeline et sauvegarder un instantane.")

    with st.form("scrape_form"):
        city_choice = st.selectbox("Ville", ["Paris", "Lyon", "Marseille"], index=0)

        city_to_slug = {
            "Paris": "paris-75000",
            "Lyon": "lyon-69000",
            "Marseille": "marseille-13000",
        }
        search_zone = city_to_slug[city_choice]

        use_filters = st.checkbox("Activer les filtres (prix / surface / pieces)", value=True)

        with st.expander("Filtres", expanded=True):
            c1, c2 = st.columns(2)

            with c1:
                min_price = st.number_input("Prix min (€)", min_value=0, value=0, step=50)
                max_price = st.number_input("Prix max (€)", min_value=0, value=0, step=50)

                min_area = st.number_input("Surface min (m²)", min_value=0, value=0, step=5)
                max_area = st.number_input("Surface max (m²)", min_value=0, value=0, step=5)

            with c2:
                min_rooms = st.number_input("Pieces min", min_value=0, value=0, step=1)
                max_rooms = st.number_input("Pieces max", min_value=0, value=0, step=1)

                min_bedrooms = st.number_input("Chambres min", min_value=0, value=0, step=1)
                max_bedrooms = st.number_input("Chambres max", min_value=0, value=0, step=1)

        start_page = st.number_input("Page de debut", min_value=1, value=1, step=1)
        end_page = st.number_input("Page de fin", min_value=1, value=1, step=1)
        scrape_submitted = st.form_submit_button("Demarrer le scraping")

    table = st.empty()
    status = st.empty()
    progress = st.progress(st.session_state.scrape_progress)

    if scrape_submitted:
        if start_page > end_page:
            st.error("Page de debut must be <= end page.")
        else:
            # Build Bienici API filter dict (only if enabled)
            user_filters = None
            if use_filters:
                user_filters = {}

                def add_if(key: str, val: int):
                    if val is not None and int(val) > 0:
                        user_filters[key] = int(val)

                # Bienici API keys: minPrice/maxPrice/minArea/maxArea/minRooms/maxRooms/minBedrooms/maxBedrooms
                add_if("minPrice", min_price)
                add_if("maxPrice", max_price)
                add_if("minArea", min_area)
                add_if("maxArea", max_area)
                add_if("minRooms", min_rooms)
                add_if("maxRooms", max_rooms)
                add_if("minBedrooms", min_bedrooms)
                add_if("maxBedrooms", max_bedrooms)

                # Basic consistency checks (only when both sides provided)
                def bad_range(lo, hi):
                    return (lo is not None and hi is not None and int(lo) > 0 and int(hi) > 0 and int(lo) > int(hi))

                if bad_range(min_price, max_price) or bad_range(min_area, max_area) or bad_range(min_rooms, max_rooms) or bad_range(min_bedrooms, max_bedrooms):
                    st.error("Erreur: un des filtres min est > au max.")
                    st.stop()

            zone_code = scrapeBi.get_zoneCode(search_zone)
            if not zone_code or not zone_code.get("zoneIds"):
                st.error("Aucun zoneId trouve pour cette zone de recherche.")
            else:
                zone_ids = zone_code["zoneIds"]

                total_pages = int(end_page - start_page + 1)
                scraped_pages = 0

                save_listings_root = Path("ressources/out_data/bienici")
                save_listings_root.mkdir(parents=True, exist_ok=True)
                save_listings_path = f"{save_listings_root}{os.sep}"

                old_path = get_latest_ds(ds_root=save_listings_root)
                old_df = pd.read_csv(old_path) if old_path is not None else pd.DataFrame()

                save_df = old_df.copy()
                show_all = pd.DataFrame()

                for page in range(int(start_page), int(end_page) + 1):
                    out = scrapeBi.scrape_page(
                        zoneIds=zone_ids,
                        page=page,
                        camembert_bundle=camembert_bundle,
                        old_listings=save_df,
                        filters=user_filters
                    )

                    page_show = out["show"]

                    if page_show.empty:
                        status.write(f"Page {page} : aucune annonce, arret.")
                        break

                    page_show = pl0.run_pipeline(df=page_show)

                    show_all = pd.concat([show_all, page_show], ignore_index=True)
                    show_all = show_all.drop_duplicates(subset=["id"], keep="last")

                    if save_df is None or save_df.empty:
                        save_df = page_show.copy()
                    else:
                        if "id" in save_df.columns:
                            save_df = save_df.drop_duplicates(subset=["id"], keep="last")
                            save_df = pd.concat(
                                [save_df[~save_df["id"].isin(page_show["id"])], page_show],
                                ignore_index=True,
                            )
                        else:
                            save_df = pd.concat([save_df, page_show], ignore_index=True)

                    scraped_pages += 1
                    st.session_state.scrape_status = (
                        f"Scraped page {page} ({len(page_show)} ads). "
                        f"Total new ads shown: {len(show_all)}"
                    )
                    st.session_state.scrape_progress = scraped_pages / total_pages
                    st.session_state.scrape_results = show_all

                    status.write(st.session_state.scrape_status)
                    progress.progress(st.session_state.scrape_progress)

                    cols_to_show = [c for c in displayed_features if c in show_all.columns]
                    table.dataframe(show_all[cols_to_show], use_container_width=True)

                scrapeBi.save_results(df=save_df, save_path=save_listings_path)

                if st.session_state.scrape_results is not None and len(st.session_state.scrape_results) > 0:
                    st.success(
                        f"Done. Newly scraped (shown): {len(st.session_state.scrape_results)} | "
                        f"Saved snapshot rows: {len(save_df)}"
                    )
                else:
                    st.warning("Aucune annonce scrapee.")

    if st.session_state.scrape_results is not None:
        cols_to_show = [c for c in displayed_features if c in st.session_state.scrape_results.columns]
        if cols_to_show:
            table.dataframe(
                st.session_state.scrape_results[cols_to_show],
                use_container_width=True,
            )
        if st.session_state.scrape_status:
            status.write(st.session_state.scrape_status)
        progress.progress(st.session_state.scrape_progress)

with right_col:
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
                st.info("No Categories de POI returned.")
