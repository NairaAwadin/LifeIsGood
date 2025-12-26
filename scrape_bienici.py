import json
import requests
import pandas as pd
import get_DistilCamemBERT as camembert


HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    ),
    "accept": "application/json, text/plain, */*",
    "x-requested-with": "XMLHttpRequest",
    "referer": "https://www.bienici.com/",
}

def get_zoneCode(slug: str):
    print(f"[SYS] : STEP 1 - Getting zone codes for {slug!r}...")
    url = f"https://res.bienici.com/suggest.json?q={slug}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERR] HTTP error when calling Bienici: {e}")
        return None
    if not data:
        print("[SYS] : No data returned from suggest.json")
        return None
    '''
    print(f"[DATA LENGTH] : {len(data)}")
    print(f"[DATA TYPE] : {type(data)}")
    print(f"[DATA CONTENT] :\n{data}")
    if len(data) > 0 :
        print("[TEST EXAMPLE] : ")
        sample_data = data[0]
        print(f"[SAMPLE TYPE] : {type(sample_data)}")
        keys = list(sample_data.keys())
        print(f"[SAMPLE KEYS] :")
        for i,k in enumerate(keys) :
            print(f"[KEY-{i}] : {k}")
    '''
    #Convert data to dataframe to analyze better.
    #df_data = pd.DataFrame(data)
    #df_data.to_csv("get_zoneIds_data.csv")
    zoneCode = {
        "zoneIds" : data[0].get("zoneIds", []),#get(key,default_value)
        "name" : data[0].get("name", "")
    }
    if not zoneCode["zoneIds"]:
        print("[SYS] : No zoneIds found in suggest response")
        return None
    #print(f"[Extracted zoneCode] : {zoneCode}")
    return zoneCode

def get_filters(
        zone_ids: list,
        page: int,
        page_size: int = 24,
        extra_filters: dict | None = None,
    ):
    print(f"[SYS] : STEP 2 - Building filters for page {page}...")

    offset = (page - 1) * page_size

    filters = {
        "size": page_size,
        "from": offset,
        "page": page,
        "onTheMarket": [True],
        "zoneIdsByTypes": {"zoneIds": zone_ids},
        "filterType": "rent",  # can be switched to buy.
        "propertyType": ["flat"],
        "sortBy": "relevance",
        "sortOrder": "desc",
    }

    # Merge user-provided filters (do NOT let UI override paging or zone ids)
    if extra_filters:
        protected = {"size", "from", "page", "zoneIdsByTypes"}
        for k, v in extra_filters.items():
            if v is None:
                continue
            if k in protected:
                continue
            filters[k] = v

    print("         Filters ready.")
    return filters


def call_bienici_api(filters: dict):
    print("[SYS] : STEP 3 - Calling Bienici realEstateAds.json API...")
    api_url = "https://www.bienici.com/realEstateAds.json"
    params = {"filters": json.dumps(filters)}
    try:
        response = requests.get(api_url, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[ERR] HTTP error when calling Bienici: {e}")
        return None
    try:
        payload = response.json()
    except ValueError:
        print("[ERR] Response is not valid JSON")
        return None
    ads = payload.get("realEstateAds")
    if ads is None:
        print("[ERR] 'realEstateAds' key missing in payload")
        return None
    if not ads:
        print("[WARN] Bienici returned 0 ads for these filters")
        return payload
    print(f"[SYS] : Received {len(ads)} ads from Bienici.")
    return payload


def get_all_keys(ad : dict, parent_key : str = ""):
    paths = set()
    for k,v in ad.items():
        if len(parent_key) > 0 :
            path = f"{parent_key}//{k}{type(ad[k])}"
        else :
            path = f"{k}{type(ad[k])}"
        paths.add(path)
        if isinstance(v,dict):
            paths.update(get_all_keys(ad=ad[k], parent_key=path))
    return paths
'''
[KEY-(0)] : district<class 'dict'>//type_id<class 'int'>
[KEY-(1)] : adTypeFR<class 'str'>
[KEY-(2)] : userRelativeData<class 'dict'>//importAccountId<class 'str'>
[KEY-(3)] : isBienIciExclusive<class 'bool'>
[KEY-(4)] : status<class 'dict'>//highlighted<class 'bool'>
[KEY-(5)] : status<class 'dict'>
[KEY-(6)] : useJuly2021EnergyPerformanceDiagnostic<class 'bool'>
[KEY-(7)] : blurInfo<class 'dict'>
[KEY-(8)] : chargingStations<class 'dict'>
[KEY-(9)] : userRelativeData<class 'dict'>//canSetAsFeatured<class 'bool'>
[KEY-(10)] : propertyType<class 'str'>
[KEY-(11)] : userRelativeData<class 'dict'>
[KEY-(12)] : chargingStations<class 'dict'>//providers<class 'list'>
[KEY-(13)] : city<class 'str'>
[KEY-(14)] : energyClassification<class 'str'>
[KEY-(15)] : price<class 'int'>
[KEY-(16)] : exposition<class 'str'>
[KEY-(17)] : displayInsuranceEstimation<class 'bool'>
[KEY-(18)] : energyValue<class 'int'>
[KEY-(19)] : userRelativeData<class 'dict'>//canSeePublicationCertificateHtml<class 'bool'>
[KEY-(20)] : greenhouseGazValue<class 'int'>
[KEY-(21)] : district<class 'dict'>//id_type<class 'int'>
[KEY-(22)] : adType<class 'str'>
[KEY-(23)] : status<class 'dict'>//is3dHighlighted<class 'bool'>
[KEY-(24)] : displayDistrictName<class 'bool'>
[KEY-(25)] : blurInfo<class 'dict'>//centroid<class 'dict'>//lon<class 'float'>
[KEY-(26)] : charges<class 'int'>
[KEY-(27)] : roomsQuantity<class 'int'>
[KEY-(28)] : postalCode<class 'str'>
[KEY-(29)] : blurInfo<class 'dict'>//centroid<class 'dict'>//lat<class 'float'>
[KEY-(30)] : blurInfo<class 'dict'>//type<class 'str'>
[KEY-(31)] : district<class 'dict'>//postal_code<class 'str'>
[KEY-(32)] : status<class 'dict'>//closedByUser<class 'bool'>
[KEY-(33)] : blurInfo<class 'dict'>//radius<class 'int'>
[KEY-(34)] : status<class 'dict'>//onTheMarket<class 'bool'>
[KEY-(35)] : descriptionTextLength<class 'int'>
[KEY-(36)] : endOfPromotedAsExclusive<class 'int'>
[KEY-(37)] : district<class 'dict'>//id_polygone<class 'int'>
[KEY-(38)] : userRelativeData<class 'dict'>//isAdmin<class 'bool'>
[KEY-(39)] : userRelativeData<class 'dict'>//accountIds<class 'list'>
[KEY-(40)] : minEnergyConsumption<class 'int'>
[KEY-(41)] : transactionType<class 'str'>
[KEY-(42)] : safetyDeposit<class 'int'>
[KEY-(43)] : departmentCode<class 'str'>
[KEY-(44)] : modificationDate<class 'str'>
[KEY-(45)] : blurInfo<class 'dict'>//position<class 'dict'>
[KEY-(46)] : userRelativeData<class 'dict'>//searchAccountIds<class 'list'>
[KEY-(47)] : chargesMethod<class 'str'>
[KEY-(48)] : id<class 'str'>
[KEY-(49)] : userRelativeData<class 'dict'>//canSeePublicationCertificatePdf<class 'bool'>
[KEY-(50)] : district<class 'dict'>//cp<class 'str'>
[KEY-(51)] : userRelativeData<class 'dict'>//canOpenAdDetail<class 'bool'>
[KEY-(52)] : pricePerSquareMeter<class 'float'>
[KEY-(53)] : postalCodeForSearchFilters<class 'str'>
[KEY-(54)] : status<class 'dict'>//isLeading<class 'bool'>
[KEY-(55)] : description<class 'str'>
[KEY-(56)] : reference<class 'str'>
[KEY-(57)] : priceHasDecreased<class 'bool'>
[KEY-(58)] : relevanceBonus<class 'int'>
[KEY-(59)] : highlightMailContact<class 'bool'>
[KEY-(60)] : blurInfo<class 'dict'>//position<class 'dict'>//lon<class 'float'>
[KEY-(61)] : addressKnown<class 'bool'>
[KEY-(62)] : phoneDisplays<class 'list'>
[KEY-(63)] : greenhouseGazClassification<class 'str'>
[KEY-(64)] : toiletQuantity<class 'int'>
[KEY-(65)] : showerRoomsQuantity<class 'int'>
[KEY-(66)] : hasElevator<class 'bool'>
[KEY-(67)] : customerId<class 'str'>
[KEY-(68)] : status<class 'dict'>//autoImported<class 'bool'>
[KEY-(69)] : newProperty<class 'bool'>
[KEY-(70)] : opticalFiberStatus<class 'str'>
[KEY-(71)] : hasTerrace<class 'bool'>
[KEY-(72)] : userRelativeData<class 'dict'>//canSeeAddress<class 'bool'>
[KEY-(73)] : accountType<class 'str'>
[KEY-(74)] : district<class 'dict'>//code_insee<class 'str'>
[KEY-(75)] : heating<class 'str'>
[KEY-(76)] : rentWithoutCharges<class 'int'>
[KEY-(77)] : energyPerformanceDiagnosticDate<class 'str'>
[KEY-(78)] : blurInfo<class 'dict'>//bbox<class 'list'>
[KEY-(79)] : userRelativeData<class 'dict'>//isAdModifier<class 'bool'>
[KEY-(80)] : district<class 'dict'>
[KEY-(81)] : floor<class 'int'>
[KEY-(82)] : userRelativeData<class 'dict'>//canSeeStats<class 'bool'>
[KEY-(83)] : nothingBehindForm<class 'bool'>
[KEY-(84)] : userRelativeData<class 'dict'>//canSeeExactPosition<class 'bool'>
[KEY-(85)] : district<class 'dict'>//id<class 'int'>
[KEY-(86)] : blurInfo<class 'dict'>//position<class 'dict'>//lat<class 'float'>
[KEY-(87)] : bedroomsQuantity<class 'int'>
[KEY-(88)] : title<class 'str'>
[KEY-(89)] : userRelativeData<class 'dict'>//canSeeRealDates<class 'bool'>
[KEY-(90)] : with3dModel<class 'bool'>
[KEY-(91)] : surfaceArea<class 'int'>
[KEY-(92)] : bathroomsQuantity<class 'int'>
[KEY-(93)] : maxEnergyConsumption<class 'int'>
[KEY-(94)] : blurInfo<class 'dict'>//centroid<class 'dict'>
[KEY-(95)] : adCreatedByPro<class 'bool'>
[KEY-(96)] : district<class 'dict'>//name<class 'str'>
[KEY-(97)] : publicationDate<class 'str'>
[KEY-(98)] : district<class 'dict'>//insee_code<class 'str'>
[KEY-(99)] : photos<class 'list'>
[KEY-(100)] : district<class 'dict'>//libelle<class 'str'>
'''

'''
#Base rent & charges
41,76,26,47,15

#convert DPE into euro per year
91,18,40,93

#cost refinement & interpretation
75,14,6,77,20,63,42

#(optional)location-based costs, external data
13,28,43,74

#keys for comfortable living / eco-comfort indexes
->energy_score (good if A–C & low kWh/m²)
->emission_score (good if A–C in GES)
->stability_score (recent DPE, new rules)
14,18,40,93,20,63,75,6,77

#Space & layout comfort
91,27,87,92,65,64

#Building / access & daily-life comfort
->accessibility_score (combination of floor & elevator)
->daylight_score (exposition, terrace)
->modernity_score (newProperty, energyClassification)
81,66,71,16,69

#Location context (for external comfort indicators)
->pollution data
->noise levels
->access to transport / amenities, etc.
13,28,43,74,98

'''
FEATURE_INDEX = {
    # =========================
    # A) Target / price fields
    # =========================
    "transactionType": 41,          # [41] rent vs other
    "price": 15,                    # [15] displayed monthly price (often all-in)
    "rentWithoutCharges": 76,       # [76] hors charges
    "charges": 26,                  # [26]
    "chargesMethod": 47,            # [47]
    "pricePerSquareMeter": 52,      # [52] useful target/feature depending on setup
    "priceHasDecreased": 57,        # [57] negotiation/overpricing signal (optional)

    # =========================
    # B) Location (coarse -> micro)
    # =========================
    "city": 13,                     # [13]
    "postalCode": 28,               # [28]
    "postalCodeForSearchFilters": 53,  # [53] sometimes cleaner than postalCode
    "departmentCode": 43,           # [43]

    # Neighborhood identifiers (best for joins & fixed effects)
    "district.code_insee": 74,      # [74] join to INSEE datasets
    "district.insee_code": 98,      # [98] alt INSEE code key
    "district.name": 96,            # [96]
    "district.libelle": 100,        # [100]

    # Approx micro-location + uncertainty (great for distance-based features later)
    "addressKnown": 61,             # [61] reliability proxy
    "blurInfo.centroid.lon": 25,    # [25]
    "blurInfo.centroid.lat": 29,    # [29]
    "blurInfo.radius": 33,          # [33] uncertainty radius
    "blurInfo.bbox": 78,            # [78] uncertainty box
    "blurInfo.type": 30,            # [30] blur type/category (if useful)

    # =========================
    # C) Size & layout (core rent drivers)
    # =========================
    "propertyType": 10,             # [10] apartment/house/etc.
    "surfaceArea": 91,              # [91]
    "roomsQuantity": 27,            # [27]
    "bedroomsQuantity": 87,         # [87]

    # Wet rooms / comfort
    "bathroomsQuantity": 92,        # [92]
    "showerRoomsQuantity": 65,      # [65]
    "toiletQuantity": 64,           # [64]

    # =========================
    # D) Building / access & daily-life comfort
    # =========================
    "floor": 81,                    # [81]
    "hasElevator": 66,              # [66] interacts strongly with floor
    "newProperty": 69,              # [69]

    # =========================
    # E) Unit comfort / livability
    # =========================
    "exposition": 16,               # [16] orientation / light proxy
    "hasTerrace": 71,               # [71] outdoor premium
    "nothingBehindForm": 83,        # [83] proxy: no vis-à-vis / open view
    "opticalFiberStatus": 70,       # [70] remote-work utility
    "with3dModel": 90,              # [90] listing/segment quality proxy (optional)

    # =========================
    # F) Energy / eco-cost & comfort (DPE/GES)
    # =========================
    "energyClassification": 14,      # [14] DPE letter
    "energyValue": 18,              # [18] kWh/m²/year (often)
    "minEnergyConsumption": 40,      # [40]
    "maxEnergyConsumption": 93,      # [93]
    "useJuly2021EnergyPerformanceDiagnostic": 6,  # [6] DPE method indicator
    "energyPerformanceDiagnosticDate": 77,        # [77] recency/validity
    "heating": 75,                  # [75] heating type proxy
    "greenhouseGazClassification": 63,  # [63] GES letter
    "greenhouseGazValue": 20,       # [20] GES numeric

    # =========================
    # G) Text fields (for deducing missing amenities/condition)
    # =========================
    "title": 88,                    # [88]
    "description": 55,              # [55]
    "descriptionTextLength": 35,     # [35]

    # =========================
    # H) Listing quality / extra proxies (optional)
    # =========================
    "photos": 99,                   # [99] use len(photos) as feature
    "isBienIciExclusive": 3,        # [3]
    "status.highlighted": 4,        # [4]
    "status.isLeading": 54,         # [54]
    "status.onTheMarket": 34,       # [34] for filtering active ads

    # =========================
    # I) IDs (optional but useful for dedup)
    # =========================
    "id": 48,                       # [48]
    "reference": 56,                # [56]
}
def deep_get(d: dict, path: str, default=None):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def get_dictValue(key : str, ad : dict):
    if key in ad:
        return ad[key]
    for v in ad.values():
        if isinstance(v, dict):
            result = get_dictValue(key, v)
            if result is not None:
                return result
    return None

def extract_vals_from_ad(keys: set[str], ad: dict):
    extracted = {}
    for k in keys:
        extracted[k] = deep_get(ad, k, default=None)
    return extracted

def extract_ads(keys: set[str], ads: list[dict]):
    return [extract_vals_from_ad(keys, ad) for ad in ads]

def update_data(
    old: pd.DataFrame | None,
    new: pd.DataFrame,
    key: str = "id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      to_show: only listings from the NEW scrape (deduped), with falseLocation carried from old when missing
      to_save: union of OLD + NEW (deduped), where NEW rows overwrite OLD rows for same id,
               and falseLocation is carried over if missing in new
    """
    new = new.copy()
    old = old.copy() if old is not None else pd.DataFrame()

    # ensure cache col exists
    if "falseLocation" not in new.columns:
        new["falseLocation"] = pd.NA
    if (not old.empty) and ("falseLocation" not in old.columns):
        old["falseLocation"] = pd.NA

    # dedup new
    if key not in new.columns:
        raise ValueError(f"update_data(): key='{key}' not found in NEW df.")
    new = new.drop_duplicates(subset=[key], keep="last")

    # if no valid old -> show=new, save=new
    if old.empty or (key not in old.columns):
        return new.copy(), new.copy()

    # dedup old
    old = old.drop_duplicates(subset=[key], keep="last")

    # index for alignment
    old_idx = old.set_index(key)
    new_idx = new.set_index(key)

    # union columns
    all_cols = sorted(set(old_idx.columns) | set(new_idx.columns))
    old_idx = old_idx.reindex(columns=all_cols)
    new_idx = new_idx.reindex(columns=all_cols)

    # ---- to_show: only NEW ids, but carry falseLocation from old if missing ----
    to_show = new_idx.copy()
    to_show["falseLocation"] = new_idx["falseLocation"].combine_first(old_idx["falseLocation"])
    to_show = to_show.reset_index()

    # ---- to_save: union OLD + NEW, prefer NEW values, but carry falseLocation ----
    to_save = new_idx.combine_first(old_idx)  # keep NEW values; fill missing with OLD
    to_save["falseLocation"] = new_idx["falseLocation"].combine_first(old_idx["falseLocation"])
    to_save = to_save.reset_index()
    out_dict = {
        "show" : to_show,
        "save" : to_save
    }
    return out_dict

def scrape_page(
    zoneIds: list,
    page: int,
    camembert_bundle=None,
    old_listings: pd.DataFrame | None = None,
    filters: dict | None = None
) -> dict:
    if not zoneIds:
        print("[SYS] : Empty zoneIds, aborting page scrape.")
        return {"show": pd.DataFrame(), "save": pd.DataFrame()}

    filters = get_filters(zone_ids=zoneIds, page=page, extra_filters=filters)
    payload = call_bienici_api(filters=filters)
    if payload is None:
        print("[SYS] : No payload returned for this page.")
        return {"show": pd.DataFrame(), "save": pd.DataFrame()}

    ads = payload.get("realEstateAds", [])
    if not ads:
        print("[SYS] : No ads on this page.")
        return {"show": pd.DataFrame(), "save": pd.DataFrame()}

    cleaned_ads = extract_ads(keys=set(FEATURE_INDEX.keys()), ads=ads)
    new = pd.DataFrame(cleaned_ads)

    # ensure cache col exists
    new["falseLocation"] = pd.NA

    # --- carry cached falseLocation from old into NEW (so we don't recompute) ---
    if old_listings is not None and (not old_listings.empty) and ("id" in old_listings.columns):
        if "falseLocation" not in old_listings.columns:
            old_listings = old_listings.copy()
            old_listings["falseLocation"] = pd.NA

        cache = (
            old_listings.drop_duplicates(subset=["id"], keep="last")
                        .set_index("id")["falseLocation"]
        )
        new["falseLocation"] = new["id"].map(cache)

    # --- only compute for missing falseLocation in the NEW scrape ---
    if camembert_bundle is not None and len(new):
        m_pred = new["falseLocation"].isna()
        if m_pred.any():
            features = ["postalCode", "city", "description"]
            new.loc[m_pred, "input_text"] = new.loc[m_pred, features].apply(
                lambda r: camembert.build_input_text(
                    postalCode=r["postalCode"],
                    city=r["city"],
                    description=r["description"],
                    desc_max_chars=2000
                ),
                axis=1
            )
            new.loc[m_pred, "falseLocation"] = new.loc[m_pred, "input_text"].apply(
                lambda x: camembert.predict(
                    input_model=camembert_bundle["model"],
                    tokenizer=camembert_bundle["tokenizer"],
                    input_text=x
                )
            )

    new = new.drop_duplicates(subset=["id"], keep="last")

    # build {show, save}
    if old_listings is None or old_listings.empty:
        to_show = new.copy()
        to_save = new.copy()
    else:
        out = update_data(old=old_listings, new=new, key="id")
        to_show = out["show"]
        to_save = out["save"]

    out_dict = {"show": to_show, "save": to_save}
    return out_dict
"""
def scrape_pages(search_zone: str, start_page: int, end_page: int | None = None) -> pd.DataFrame:

    results = pd.DataFrame()

    zoneCode = get_zoneCode(search_zone)
    if not zoneCode or not zoneCode.get("zoneIds"):
        print("[SYS] : Cannot scrape pages without valid zoneIds.")
        return results

    zoneIds = zoneCode["zoneIds"]

    if end_page is not None:
        if start_page > end_page:
            print("[SYS] ERR start_page has to be <= end_page")
            return results
        page = start_page
        while page <= end_page:
            page_df = scrape_page(zoneIds=zoneIds, page=page)
            if page_df.empty:
                break
            results = pd.concat([results, page_df], ignore_index=True)
            page += 1
    else:
        page = start_page
        while True:
            page_df = scrape_page(zoneIds=zoneIds, page=page)
            if page_df.empty:
                break
            results = pd.concat([results, page_df], ignore_index=True)
            page += 1
    results = results.drop_duplicates()
    return results
"""
def save_results(df: pd.DataFrame, save_path : str = "ressources/data_out/bienici"):
    print(len(df))
    df.to_csv(f"{save_path}bienIci_listings.csv", index=None)

def main() :
    #pages = scrape_pages(search_zone="75000", start_page=1, end_page=110)
    #print(f"Scraped {len(pages)} ads.")
    #save_results(pages=pages)
    return
if __name__ == "__main__":
    main()
