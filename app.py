import streamlit as st
import get_DistilCamemBERT as camembert
import pipeline_0 as pl0
import scrape_bienici as scrapeBi
from pathlib import Path
import os
import pandas as pd
import get_catboost as cat
import pipeline_v as pv
from gestion_donnees import create_dictionnaire_recherche
from calculs_transports import (
    cout_transports_publics,
    cout_voiture,
    budget_logement
)
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


displayed_features = [
    "id",
    "price",
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

st.set_page_config(
    page_title="LifeIsGood",
    page_icon="🏡"
)

st.title("LifeIsGood : Assistant Recherche de Logement")
st.write("---")

if "scrape_results" not in st.session_state:
    st.session_state.scrape_results = None
if "scrape_status" not in st.session_state:
    st.session_state.scrape_status = ""
if "scrape_progress" not in st.session_state:
    st.session_state.scrape_progress = 0.0

if 'page' not in st.session_state:
    st.session_state.page = "accueil"

def accueil():
    st.header("Bienvenue dans votre application assistant logement.")
    st.write("On est pour vous aider à trouver un logement dans 3 villes pour le moment : Paris, Lyon et Marseille.")

    col_1, col_2 = st.columns(2)

    with col_1:
        if st.button("**Je cherche un logement**", use_container_width=True, type="primary"):
            st.session_state.page = "recherche"
            st.rerun()
    with col_2:
        if st.button("**Je veux estimer mon logement**", use_container_width=True):
            st.session_state.page = "estimation"
            st.rerun()

    st.markdown("""
    ### Comment ça fonctionne ?

    1. **Je cherche un logement** :
       - Calculez vos dépenses de transport
       - Déterminez votre budget logement optimal
       - Obtenez des recommandations personnalisées

    2. **Je veux estimer mon logement** (bientôt disponible)
       - Estimez la valeur de votre bien immobilier
    """)

def recherche_logement():
    st.header("Recherche de logement")

    # Bouton retour
    if st.button("Retour à l'accueil"):
        st.session_state.page = "accueil"
        st.rerun()

    st.write("Remplissez le formulaire ci-dessous pour commencer votre recherche.")

    #formulaire
    ville = st.selectbox("Dans quelle ville cherchez-vous ?", ["Paris", "Lyon", "Marseille"])
    
    #----scrape id mika
    city_to_slug = {
        "Paris": "paris-75000",
        "Lyon": "lyon-69000",
        "Marseille": "marseille-13000",
    }
    search_zone = city_to_slug[ville]

    #----
    st.write(f"Vous avez choisi : {ville}")

    st.write("---")
    choix_financier = st.radio(
        "**Vous souhaitez indiquer :**",
        ["Votre salaire mensuel net", "Votre budget logement mensuel"],
        index=None,
    )

    if choix_financier == "Votre salaire mensuel net":
        salaire = st.slider("Votre salaire mensuel net (€)", 0, 10000, 2000)
        st.write(f"Votre salaire : {salaire}€")
        budget_log = None
    else:
        budget_log = st.slider("Votre budget logement mensuel (€)", 0, 5000, 800)
        st.write(f"Votre budget logement : {budget_log}€")
        salaire = None

    cout_transport = 0
    pourcentage_transport = 0
    consommation_saisie = None

    st.write("---")
    st.write("**Informations transport :**")

    if salaire is not None:
        st.write("Puisque vous avez indiqué votre salaire, nous allons calculer vos dépenses de transport.")

        type_transport = st.radio(
            "**Type de transport principal :**",
            ["Transport public", "Voiture"],
            horizontal=True
        )

        if type_transport == "Transport public":
            profil = st.selectbox(
                "**Votre profil :**",
                ["Adulte", "Etudiant", "Senior"]
            )
            st.write(f"Profil sélectionné : {profil}")

            try:
                cout_transport, pourcentage_transport = cout_transports_publics(ville, profil, salaire)
                st.success(f"Coût transport : {cout_transport}€ ({pourcentage_transport:.1f}% du salaire)")
            except Exception as e:
                st.error(f"Erreur dans le calcul: {str(e)}")
                cout_transport, pourcentage_transport = 0, 0
        else:
            type_voiture = st.selectbox(
                "**Type de voiture :**",
                ["Essence", "Diesel", "Electrique"]
            )

            km_mensuel = st.slider(
                "Kilométrage mensuel estimé (km) :",
                0, 10000, 1000
            )
            st.write(f"Kilométrage : {km_mensuel} km/mois")

            if type_voiture in ["Essence", "Diesel"]:
                connait_consommation = st.checkbox("Je connais la consommation de mon véhicule")
                if connait_consommation:
                    consommation_saisie = st.number_input(
                        "Consommation (L/100km) :",
                        value=6,
                        placeholder="Consommation de votre véhicule"
                    )
                    st.write(f"Consommation : {consommation_saisie} L/100km")
                else:
                    st.info("Valeur par défaut utilisée : 6L/100km")
                    consommation_saisie = 6
            else:  # Électrique
                connait_consommation = st.checkbox("Je connais la consommation de mon véhicule électrique")
                if connait_consommation:
                    consommation_saisie = st.number_input(
                        "Consommation (kWh/100km) :",
                        value=None,
                        placeholder="Consommation de votre véhicule électrique",
                        help="Moyenne : 17 kWh/100km"
                    )
                    st.write(f"Consommation : {consommation_saisie} kWh/100km")
                else:
                    st.info("Valeur par défaut utilisée : 17kWh/100km")
                    consommation_saisie = 17

            try:
                cout_transport, pourcentage_transport = cout_voiture(
                    ville, type_voiture, km_mensuel, salaire, consommation_saisie
                )
                st.success(f"Coût voiture : {cout_transport}€ ({pourcentage_transport:.1f}% du salaire)")
            except Exception as e:
                st.error(f"Erreur dans le calcul: {str(e)}")
                cout_transport, pourcentage_transport = 0, 0
    else:
        st.info("Avec un budget logement prédéfini, nous ne calculons pas les dépenses de transport.")



    #criteres logement
    st.write("---")
    st.write("**Critères du logement :**")

    col_1, col_2 = st.columns(2)
    use_filters = st.checkbox("Activer les filtres", value=True)
    with col_1:
        surface_min = st.number_input(
            "Surface minimum (m²) :",
            min_value=0,
            value=0,
            step=5
        )
        #st.write(f"Surface minimum : {surface_min} m²")
        min_rooms = st.number_input("Pieces min", min_value=0, value=0, step=1)
        min_bedrooms = st.number_input("Chambres min", min_value=0, value=0, step=1)
        min_price = st.number_input("Prix min (EUR)", min_value=0, value=0, step=50)

    with col_2:
        surface_max = st.number_input(
            "Surface maximum (m²) :",
            min_value=surface_min,
            value=0,
            step=5
        )
        #st.write(f"Surface maximum : {surface_max} m²")
        max_rooms = st.number_input("Pieces max", min_value=0, value=0, step=1)
        max_bedrooms = st.number_input("Chambres max", min_value=0, value=0, step=1)
        preference = st.text_input("Preference", value = "", placeholder = "Indiquez vos préférences")

        #st.write(f"Nombre de pieces maximum : {max_rooms}")
        #st.write(f"Nombre de chambres maximum : {max_bedrooms}")

    st.write("---")
    start_page = st.number_input("Page de debut", min_value=1, value=1, step=1)
    end_page = st.number_input("Page de fin", min_value=1, value=1, step=1)
    if st.button("Lancer la recherche", type="primary", use_container_width=True):
        params = {
            "ville": ville,
            "min_price" : min_price,
            "search_zone" : search_zone,
            "use_filters" : use_filters,
            "surface_min": surface_min,
            "surface_max": surface_max,
            "min_chambres": min_bedrooms,
            "max_chambres": max_bedrooms,
            "min_pieces" :min_rooms,
            "max_pieces" :max_rooms,
            "preference" : preference,
            "start_page" : start_page,
            "end_page" : end_page
        }

        if salaire is not None:
            budget_recommande = budget_logement(salaire, pourcentage_transport)
            params.update({
                "salaire": salaire,
                "type_transport": type_transport,
                "cout_transport": cout_transport,
                "pourcentage_transport": pourcentage_transport,
                "budget_recommande": budget_recommande
            })

            if type_transport == "Transport public":
                params["profil"] = profil
            else:
                params.update({
                    "type_voiture": type_voiture,
                    "km_mensuel": km_mensuel,
                    "consommation": consommation_saisie
                })
        else:
            params["budget_logement"] = budget_log

        donnees = create_dictionnaire_recherche(params)
        st.session_state["donnees_recherche"] = donnees
        display_compte_rendu(donnees)

def display_compte_rendu(donnees):
    st.write("---")
    st.header("Résultats de votre recherche")
    st.write("**Informations générales**")

    col_1, col_2 = st.columns(2)

    with col_1:
        st.write(f"**Ville :** {donnees.get('ville', 'Non spécifiée')}")
        st.write(f"**Surface minimale :** {donnees.get('surface_min', 0)} m²")

    with col_2:
        st.write(f"**Surface maximale :** {donnees.get('surface_max', 0)} m²")

    st.write("---")
    mode = donnees.get("mode")
    if mode == "salaire":
        st.write("**Mode : Calcul basé sur le salaire**")
        st.write("---")

        st.subheader("Analyse financière")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Salaire mensuel", f"{donnees.get('salaire', 0)}€")

        with col_b:
            st.metric("Coût transport", f"{donnees.get('cout_transport', 0)}€")

        with col_c:
            st.metric("% du salaire", f"{donnees.get('pourcentage_transport', 0)}%")
        # Détails du transport
        st.write("**Détails transport :**")

        type_transport = donnees.get("type_transport")
        if type_transport == "Transport public":
            st.write(f"• **Type :** Transport public")
            st.write(f"• **Profil :** {donnees.get('profil', 'Non spécifié')}")
        elif type_transport == "Voiture":
            st.write(f"• **Type :** Voiture")
            st.write(f"• **Type de voiture :** {donnees.get('type_voiture', 'Non spécifié')}")
            st.write(f"• **Kilométrage mensuel :** {donnees.get('km_mensuel', 0)} km")
            if donnees.get("consommation"):
                if donnees.get("type_voiture") in ["Essence", "Diesel"]:
                    st.write(f"• **Consommation :** {donnees.get('consommation')} L/100km")
                else:
                    st.write(f"• **Consommation :** {donnees.get('consommation')} kWh/100km")

        st.write("---")

        st.subheader("🏡 Budget logement recommandé")

        budget_recommande = donnees.get("budget_recommande", 0)

        # Afficher le budget
        st.success(f"**{budget_recommande}€ / mois**")

        pourcentage_transport = donnees.get("pourcentage_transport", 0)
        pourcentage_logement = donnees.get("pourcentage_logement", 40)
        pourcentage_total = pourcentage_transport + pourcentage_logement

        # Affichage des pourcentages
        st.write("**Répartition du budget :**")

        col_p1, col_p2, col_p3 = st.columns(3)

        with col_p1:
            st.metric("Transport", f"{pourcentage_transport}%")

        with col_p2:
            st.metric("Logement", f"{pourcentage_logement}%")

        with col_p3:
            st.metric("Total", f"{pourcentage_total}%")

        # Objectif
        #st.write(f"**Objectif :** Transport + Logement = 40% du salaire")

        # Évaluation
       # st.write("**Évaluation :**")

        if pourcentage_total <= 40:
            st.success("**Excellent !** Votre budget est bien équilibré.")
        elif pourcentage_total <= 50:
            st.warning("**Attention :** Votre budget est un peu élevé.")
        else:
            st.error("**Alerte :** Votre budget est trop élevé par rapport à vos revenus.")


    #display scraped data -mika
    table = st.empty()
    status = st.empty()
    progress = st.progress(st.session_state.scrape_progress)
    start_page = donnees.get("start_page")
    end_page = donnees.get("end_page")
    use_filters = donnees.get("use_filters")
    min_price = donnees.get("min_price")
    max_price = donnees.get("max_price")
    surface_min = donnees.get("surface_min")
    surface_max = donnees.get("surface_max")
    min_rooms = donnees.get("min_pieces")
    max_rooms = donnees.get("max_pieces")
    min_bedrooms = donnees.get("min_chambres")
    max_bedrooms = donnees.get("max_chambres")
    search_zone = donnees.get("search_zone")

    budget_logement = donnees.get("budget_logement", 0)
    budget_recommande = donnees.get("budget_recommande", 0)
    if max_price is None:
        max_price = max(budget_logement or 0, budget_recommande or 0)
    if start_page > end_page:
            st.error("La page de debut doit etre <= a la page de fin.")
    else:
        user_filters = {}
        def add_if(key: str, val: int):
            if val is not None and int(val) > 0:
                user_filters[key] = int(val)
        add_if("maxPrice", max_price)
        if use_filters:



            add_if("minPrice", min_price)
            add_if("minArea", surface_min)
            add_if("maxArea", surface_max)
            add_if("minRooms", min_rooms)
            add_if("maxRooms", max_rooms)
            add_if("minBedrooms", min_bedrooms)
            add_if("maxBedrooms", max_bedrooms)

            def bad_range(lo, hi):
                return (lo is not None and hi is not None and int(lo) > 0 and int(hi) > 0 and int(lo) > int(hi))

            if bad_range(min_price, max_price) or bad_range(surface_min, surface_max) or bad_range(min_rooms, max_rooms) or bad_range(min_bedrooms, max_bedrooms):
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
            if not old_df.empty:
                old_df = pl0.run_pipeline(df=old_df)
            show_all = pd.DataFrame()
            save_all = old_df.copy()

            for page in range(int(start_page), int(end_page) + 1):
                out = scrapeBi.scrape_page(
                    zoneIds=zone_ids,
                    page=page,
                    camembert_bundle=camembert_bundle,
                    old_listings=save_all,
                    filters=user_filters if len(user_filters) > 0 else None,
                )

                page_show = out["show"]
                page_save = out["save"]

                if page_show.empty:
                    status.write(f"Page {page} : aucune annonce, arret.")
                    break

                page_show = pl0.run_pipeline(df=page_show)
                page_save = pl0.run_pipeline(df=page_save)
                show_all = pd.concat([show_all, page_show], ignore_index=True)
                show_all = show_all.drop_duplicates(subset=["id"], keep="last")

                if save_all is None or save_all.empty:
                    save_all = page_show.copy()
                else:
                    save_all = pd.concat([save_all, page_save], ignore_index=True)
                    save_all = save_all.drop_duplicates(subset=["id"], keep="last")

                scraped_pages += 1
                st.session_state.scrape_status = (
                    f"Page {page} scrapee ({len(page_show)} annonces). "
                    f"Total nouvelles annonces affichees : {len(show_all)}"
                )
                st.session_state.scrape_progress = scraped_pages / total_pages
                st.session_state.scrape_results = show_all

                status.write(st.session_state.scrape_status)
                progress.progress(st.session_state.scrape_progress)

                display_df = show_all.copy()
                if "falseLocation" in display_df.columns:
                    display_df["falseLocation"] = display_df["falseLocation"].astype(str)
                cols_to_show = [c for c in displayed_features if c in display_df.columns]
                table.dataframe(display_df[cols_to_show], use_container_width=True)

            scrapeBi.save_results(df=save_all, save_path=save_listings_path)

            if st.session_state.scrape_results is not None and len(st.session_state.scrape_results) > 0:
                st.success(
                    f"Termine. Nouvelles annonces (affichees) : {len(save_all) - len(old_df)} | "
                    f"Annonces sauvegardees : {len(save_all)}"
                )
            else:
                st.warning("Aucune annonce scrapee.")


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
def estimation_logement():
    st.header("Estimation de logement")

    # Bouton retour
    if st.button("Retour à l'accueil"):
        st.session_state.page = "accueil"
        st.rerun()

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

    #st.info("Cette fonctionnalité sera disponible prochainement !")

if st.session_state.page == "accueil":
    accueil()
elif st.session_state.page == "recherche":
    recherche_logement()
elif st.session_state.page == "estimation":
    estimation_logement()
