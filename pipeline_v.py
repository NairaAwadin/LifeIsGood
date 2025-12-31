import pandas as pd
import numpy as np
TARGET = "computedPricePerSquareMeter"  # you build this from advertisedPrice / etc
REQUIRED = [
    # to avoid mixing sale vs rent if your dataset contains both
    "transactionType",

    # location (at least these two)
    "computedPostalCode",

    # fundamentals (the core price drivers)
    "computedSurfaceArea",
    "propertyType",
    "roomsQuantity",
    "computedBedroomsQuantity",
    "computedPricePerSquareMeter"
]
NUM_COLS = [
 'blurInfo.centroid.lat',
 'blurInfo.centroid.lon',
 'computedSurfaceArea',
 'roomsQuantity',
 'computedBedroomsQuantity',
 'floor',
 'computedBathroomsQuantity',
 'computedShowerRoomsQuantity',
 'computedToiletQuantity',
 'computedEnergyValue',
 'computedMinEnergyConsumption',
 'computedMaxEnergyConsumption',
 'greenhouseGazValue'
]
BOOL_COLS = [
 'computedNewProperty',
 'computedHasElevator',
 'computedHasTerrace'
]
CAT_COLS = [
    'computedPostalCode',
    'transactionType',
    'city',
    'propertyType',
    'heating',
    'exposition',
    'computedEnergyClassification',
    'greenhouseGazClassification'
]
FEATURES = CAT_COLS + BOOL_COLS + NUM_COLS
RENAME_MAP = {
    "transactionType": "transaction_type",
    "computedPostalCode": "postal_code",
    "city": "city",
    "blurInfo.centroid.lat": "lat",
    "blurInfo.centroid.lon": "lon",
    "propertyType": "property_type",
    "computedSurfaceArea": "surface_area",
    "roomsQuantity": "rooms",
    "computedBedroomsQuantity": "bedrooms",
    "floor": "floor",
    "computedNewProperty": "is_new",
    "computedHasElevator": "has_elevator",
    "computedHasTerrace": "has_terrace",
    "heating": "heating",
    "exposition": "exposition",
    "computedBathroomsQuantity": "bathrooms",
    "computedShowerRoomsQuantity": "shower_rooms",
    "computedToiletQuantity": "toilets",
    "computedEnergyClassification": "energy_class",
    "computedEnergyValue": "energy_value",
    "computedMinEnergyConsumption": "energy_min",
    "computedMaxEnergyConsumption": "energy_max",
    "greenhouseGazClassification": "ghg_class",
    "greenhouseGazValue": "ghg_value",
    "computedPricePerSquareMeter": "price_per_sqm",  # target
}
def return_REQUIRED():
    return REQUIRED
def return_TARGET():
    return TARGET
def return_FEATURES():
    return FEATURES
def return_RENAME_MAP():
    return RENAME_MAP
def return_CAT():
    return CAT_COLS
def return_NUM():
    return NUM_COLS
def return_BOOL():
    return BOOL_COLS

def prepareX(X:pd.DataFrame,num_cols:list = NUM_COLS,
             cat_cols:list = CAT_COLS, bool_cols:list = BOOL_COLS):
    default_x_cols = num_cols + cat_cols + bool_cols
    missings = set(default_x_cols) - set(X.columns)
    if len(missings) > 0 :
        for c in missings :
            if c in num_cols :
                X[c] = np.nan
            elif c in cat_cols :
                X[c] = "__UNKNOWN__"
            elif c in bool_cols :
                X[c] = False
    X = X[default_x_cols].copy()
    for c in bool_cols :
        X[c] = X[c].astype("boolean")
        X[c] = X[c].fillna(False)
    for c in num_cols :
        X[c] = pd.to_numeric(X[c],errors="coerce")
    for c in cat_cols :
        if c == "computedPostalCode" :
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("Int64").astype("string")
        X[c] = X[c].astype("string")
        X[c] = X[c].fillna("__UNKNOWN__")
    return X
def prepareY(y:pd.Series):
    y = y.copy()
    return pd.to_numeric(y, errors="coerce")
def prepareDataset(df: pd.DataFrame, num_cols:list = NUM_COLS,
             cat_cols:list = CAT_COLS, bool_cols:list = BOOL_COLS,
                   TARGET: str = TARGET, REQUIRED: list = REQUIRED,
                  RENAME_MAP : dict = RENAME_MAP):
    # keep only rows where REQUIRED are non-NA
    m_required = df[REQUIRED].notna().all(axis=1)
    out = df.loc[m_required].copy()

    default_x_cols = num_cols + cat_cols + bool_cols

    #casting
    out = pd.concat([prepareX(X = out[default_x_cols],
                 num_cols = num_cols,
                 cat_cols = cat_cols,
                 bool_cols = bool_cols),prepareY(y = out[TARGET])],axis = 1)
    #rename (normalize cols name)
    out = out.rename(columns=RENAME_MAP)
    renamed_features = [RENAME_MAP.get(c, c) for c in FEATURES]
    renamed_target = RENAME_MAP.get(TARGET, TARGET)
    X = out[renamed_features].copy()
    y = pd.to_numeric(out[renamed_target], errors="coerce")
    return X, y
