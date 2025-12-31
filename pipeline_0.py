import numpy as np
import pandas as pd
from pathlib import Path

REQUIRED_SCHEMA = {
    "transactionType": "str",        # filter_transaction_type
    "propertyType": "str",           # filter_property_type
    "price": "numeric",
    "charges": "numeric",
    "rentWithoutCharges": "numeric",  # computedPrice
    "surfaceArea": "numeric",
    "pricePerSquareMeter": "numeric",      # computedSurfaceArea/PricePerSquareMeter
    "postalCode": "numeric",
    "postalCodeForSearchFilters": "numeric",
    "city": "str",  # computedPostalCode
    "roomsQuantity": "numeric",
    "bedroomsQuantity": "numeric",       # compute_layout
    "bathroomsQuantity": "numeric",
    "showerRoomsQuantity": "numeric",
    "toiletQuantity": "numeric",  # compute_sanitary
    "priceHasDecreased": "boolean",      # computedPriceHasDecreased (bool-like)
    "newProperty": "boolean",
    "energyValue": "numeric",
    "energyClassification": "str",     # compute_energy
    "minEnergyConsumption": "numeric",
    "maxEnergyConsumption": "numeric",  # energy consumption range fill
    "hasTerrace": "str",   # raw CSV has object, will be coerced downstream
    "hasElevator": "str",  # raw CSV has object, will be coerced downstream
}
DPE_MIDPOINT_KWHEP_M2Y = {
    "A": 35.0,
    "B": 90.0,
    "C": 145.0,
    "D": 215.0,
    "E": 290.0,
    "F": 375.0,
    "G": 450.0
}
DPE_RANGE_KWHEP_M2Y = {
    "A": (0, 70),
    "B": (70, 110),
    "C": (110, 180),
    "D": (180, 250),
    "E": (250, 330),
    "F": (330, 420),
    "G": (420, float("inf")),
}

def filter_transaction_type(df: pd.DataFrame, ttype: str = "rent") -> pd.DataFrame:
    return df.loc[df["transactionType"] == ttype].copy()
def filter_property_type(df:pd.DataFrame, ttype: str = "flat")->pd.DataFrame:
    return df.loc[df["propertyType"] == ttype].copy()

def get_computedPrice(df : pd.DataFrame) -> pd.DataFrame:
    """
    Rules :
    
    let:
    p = price
    r = rentWithoutCharges
    c = charges
    s = r + c
    es = computedPrice
    
    logics :
    if p doesn't exist, c/r doesn't exist :
    -> drop
    if p doesn't exist, c and r exist :
    -> es = s
    if p and s exist :
        p > s : es = p
        p < s : es = s
    """
    out = df.copy()
    p = out["price"].copy()
    c = out["charges"].copy()
    r = out["rentWithoutCharges"].copy()
    #drop rows where we cannot build any price
    m = p.isna() & (c.isna() | r.isna())
    out = out.loc[~m].copy()
    p = out["price"].copy()
    s = (out["charges"] + out["rentWithoutCharges"]).copy()
    # impute for missing price and take max when both exist
    es = p.where(p.notna(), s)
    both = p.notna() & s.notna()
    es = es.where(~both, np.maximum(p, s))
    #rename
    out["computedPrice"] = es
    return out

def get_computedSurfaceArea(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    m_drop = out["surfaceArea"].isna() & (out["pricePerSquareMeter"].isna() | out["computedPrice"].isna())
    out = out.loc[~m_drop].copy()

    out["computedSurfaceArea"] = out["surfaceArea"]  # copy first

    m_missing = out["computedSurfaceArea"].isna()
    out.loc[m_missing, "computedSurfaceArea"] = (out.loc[m_missing, "computedPrice"] / out.loc[m_missing, "pricePerSquareMeter"]).round()

    # ensure numeric; keep as float to avoid casting issues
    out["computedSurfaceArea"] = pd.to_numeric(out["computedSurfaceArea"], errors="coerce")
    return out

def get_computedPricePerSquareMeter(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    m_drop = out["pricePerSquareMeter"].isna() & (out["surfaceArea"].isna() | out["computedPrice"].isna())
    out = out.loc[~m_drop].copy()

    out["computedPricePerSquareMeter"] = out["pricePerSquareMeter"]  # copy first

    m_missing = out["computedPricePerSquareMeter"].isna()
    out.loc[m_missing, "computedPricePerSquareMeter"] = (out.loc[m_missing, "computedPrice"] / out.loc[m_missing, "surfaceArea"]).round(2)
    return out

def get_computedPriceHasDecreased(df : pd.DataFrame) -> pd.DataFrame:
    """
    Docstring for get_computedPriceHasDecreased
    
    :param df: Description
    :type df: pd.DataFrame
    :return: Description
    :rtype: DataFrame
    """
    out = df.copy()
    out["computedPriceHasDecreased"] = out["priceHasDecreased"].fillna(False)
    return out

def get_computedPostalCode(df : pd.DataFrame) -> pd.DataFrame :
    """
    Fill postalCode from postalCodeForSearchFilters; drop rows with no postalCode and no city.
    """
    out = df.copy()
    out["computedPostalCode"] = out["postalCode"].fillna(out["postalCodeForSearchFilters"]).copy()
    out["computedPostalCode"] = out["computedPostalCode"].astype("string")
    return out.loc[~(out[["postalCode", "city"]].isna().all(axis=1))].copy()

def compute_layout(df : pd.DataFrame)->pd.DataFrame :
    """
    Docstring for compute_layout
    
    :param df: Description
    :type df: pd.DataFrame
    :return: Description
    :rtype: DataFrame
    """
    out = df.copy()
    #drop rows with roomsQuantity == N/A
    m_rooms_ok = out["roomsQuantity"].notna() & (out["roomsQuantity"] > 0)
    out = out.loc[m_rooms_ok].copy()
    #compute/impute bedrooms
    out["computedBedroomsQuantity"] = out["bedroomsQuantity"]    
    out.loc[out["bedroomsQuantity"].isna(),"computedBedroomsQuantity"] = out.loc[out["bedroomsQuantity"].isna(),"roomsQuantity"] - 1
    return out

def compute_sanitary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute between bathroomsQuantity and showerRoomsQuantity; fill toiletQuantity with 1.
    """
    out = df.copy()
    out["computedBathroomsQuantity"] = out["bathroomsQuantity"]
    out["computedShowerRoomsQuantity"] = out["showerRoomsQuantity"]

    m_b_na = out["computedBathroomsQuantity"].isna()
    m_s_na = out["computedShowerRoomsQuantity"].isna()

    out.loc[m_s_na & ~m_b_na, "computedShowerRoomsQuantity"] = out.loc[m_s_na & ~m_b_na, "bathroomsQuantity"]
    out.loc[m_b_na & ~m_s_na, "computedBathroomsQuantity"] = out.loc[m_b_na & ~m_s_na, "showerRoomsQuantity"]

    out["computedToiletQuantity"] = out["toiletQuantity"].fillna(1)
    return out

def compute_energy(df:pd.DataFrame):
    """
    Docstring for compute_energy
    
    :param df: Description
    :type df: pd.DataFrame
    """
    out = df.copy()
    out["energyClassification"] = out["energyClassification"].astype("string").str.strip().str.upper()
    out["energyValue"] = pd.to_numeric(out["energyValue"], errors="coerce")

    out["computedEnergyValue"] = out["energyValue"]
    out["computedEnergyClassification"] = out["energyClassification"]
    out["computedEnergyValue"] = out["computedEnergyValue"].astype("Float64")
    out["computedMinEnergyConsumption"] = out["minEnergyConsumption"].astype("Float64")
    out["computedMaxEnergyConsumption"] = out["maxEnergyConsumption"].astype("Float64")
    m_energyValNa = out["computedEnergyValue"].isna()
    m_energyClassificationNa = out["computedEnergyClassification"].isna()
    #compute/impute eneryValue
    out.loc[m_energyValNa & ~m_energyClassificationNa, "computedEnergyValue"] = (
        out.loc[m_energyValNa & ~m_energyClassificationNa, "computedEnergyClassification"].apply(
            lambda x : DPE_MIDPOINT_KWHEP_M2Y[x] 
                    if isinstance(x,str) and x in DPE_MIDPOINT_KWHEP_M2Y
                    else pd.NA
        )
    )
    #compute/impute m_energyClassification

    def get_energyClassification(val):
        if val is None or pd.isna(val):
            return pd.NA
        for c, (lo,hi) in DPE_RANGE_KWHEP_M2Y.items():
            if lo <= val < hi :
                return c
        return pd.NA
    out.loc[~m_energyValNa & m_energyClassificationNa,"computedEnergyClassification"] = (
        out.loc[~m_energyValNa & m_energyClassificationNa,"computedEnergyValue"].apply(
            get_energyClassification
        )
    )
    return out

def get_computedEnergyConsumption(df : pd.DataFrame):
    out = df.copy()
    def get_range_energy(ec, s):
        if ec not in DPE_RANGE_KWHEP_M2Y or pd.isna(s):
            return (pd.NA, pd.NA)

        lo, hi = DPE_RANGE_KWHEP_M2Y[ec]
        lo_kwh = lo * s
        if lo == 0 :
            lo_kwh = pd.NA
        if hi == float("inf"):
            hi_kwh = pd.NA   # or set a cap like 500*s
        else:
            hi_kwh = hi * s
        return (lo_kwh, hi_kwh)
    
    out["computedMinEnergyConsumption"] = out["minEnergyConsumption"]
    out["computedMaxEnergyConsumption"] = out["maxEnergyConsumption"]

    m_minNa = out["minEnergyConsumption"].isna()
    m_maxNa = out["maxEnergyConsumption"].isna()
    m_ecNa = out["computedEnergyClassification"].isna()
    m_surfaceNa = out["computedSurfaceArea"].isna()

    out["computedMinEnergyConsumption"] = out["computedMinEnergyConsumption"].astype("Float64")
    out["computedMaxEnergyConsumption"] = out["computedMaxEnergyConsumption"].astype("Float64")

    out.loc[m_minNa & ~m_ecNa & ~m_surfaceNa, "computedMinEnergyConsumption"] = (
        out.loc[m_minNa & ~m_ecNa & ~m_surfaceNa].apply(
            lambda r : get_range_energy(ec=r["computedEnergyClassification"],
                                                  s=r["computedSurfaceArea"])[0],
            axis = 1
        ).astype("Float64")
    )
    out.loc[m_maxNa & ~m_ecNa & ~m_surfaceNa, "computedMaxEnergyConsumption"] = (
        out.loc[m_maxNa & ~m_ecNa & ~m_surfaceNa].apply(
            lambda r : get_range_energy(ec=r["computedEnergyClassification"],
                                                  s=r["computedSurfaceArea"])[-1],
            axis = 1
        ).astype("Float64")
    )
    out["computedAvgEnergyConsumption"] = (out["computedMinEnergyConsumption"] + out["computedMaxEnergyConsumption"]) / 2
    m = out["computedAvgEnergyConsumption"].isna() & out["computedEnergyClassification"].notna()
    out.loc[m,"computedAvgEnergyConsumption"] = out["computedEnergyClassification"].apply(lambda x : DPE_MIDPOINT_KWHEP_M2Y[x.strip().upper()]
                                                                                          if isinstance(x,str) and x in DPE_MIDPOINT_KWHEP_M2Y
                                                                                            else pd.NA  )
    return out

def get_computedHasElevator(df : pd.DataFrame):
    out = df.copy()
    col = out["hasElevator"].astype(str).str.strip().str.lower()
    true_vals = {"true", "1", "yes", "y", "oui", "vrai"}
    false_vals = {"false", "0", "no", "n", "non", "faux"}
    col = col.map(lambda x: True if x in true_vals else False if x in false_vals else pd.NA)
    col = col.astype("boolean")
    out["computedHasElevator"] = col.mask(col.isna(), False)
    return out

def get_computedNewProperty(df: pd.DataFrame):
    out = df.copy()
    col = out["newProperty"].astype(str).str.strip().str.lower()
    true_vals = {"true", "1", "yes", "y", "oui", "vrai"}
    false_vals = {"false", "0", "no", "n", "non", "faux"}
    col = col.map(lambda x: True if x in true_vals else False if x in false_vals else pd.NA)
    col = col.astype("boolean")
    out["computedNewProperty"] = col.mask(col.isna(), False)
    return out

def get_computedHasTerrace(df : pd.DataFrame):
    out = df.copy()
    col = out["hasTerrace"].astype(str).str.strip().str.lower()
    true_vals = {"true", "1", "yes", "y", "oui", "vrai"}
    false_vals = {"false", "0", "no", "n", "non", "faux"}
    col = col.map(lambda x: True if x in true_vals else False if x in false_vals else pd.NA)
    col = col.astype("boolean")
    out["computedHasTerrace"] = col.mask(col.isna(), False)
    return out

def verify_required_schema(df: pd.DataFrame) -> None:
    """Ensure required columns exist and have compatible dtypes."""
    missing = [c for c in REQUIRED_SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    type_errors = []
    for col, kind in REQUIRED_SCHEMA.items():
        s = df[col]
        if kind == "numeric":
            if not pd.api.types.is_numeric_dtype(s):
                try:
                    df[col] = pd.to_numeric(s, errors="coerce")
                    s = df[col]
                except Exception:
                    type_errors.append(f"{col} expected numeric, got {s.dtype}")
            if not pd.api.types.is_numeric_dtype(s):
                type_errors.append(f"{col} expected numeric, got {s.dtype}")
        elif kind == "str":
            if not pd.api.types.is_string_dtype(s):
                try:
                    df[col] = s.astype("string")
                    s = df[col]
                except Exception:
                    type_errors.append(f"{col} expected string, got {s.dtype}")
            if not pd.api.types.is_string_dtype(s):
                type_errors.append(f"{col} expected string, got {s.dtype}")
        elif kind == "boolean":
            if not pd.api.types.is_bool_dtype(s):
                try:
                    df[col] = s.astype("boolean")
                    s = df[col]
                except Exception:
                    type_errors.append(f"{col} expected boolean, got {s.dtype}")
            if not pd.api.types.is_bool_dtype(s):
                type_errors.append(f"{col} expected boolean, got {s.dtype}")
    if type_errors:
        raise ValueError("Schema issues: " + "; ".join(type_errors))

def run_pipeline(df : pd.DataFrame, save : bool = False, fname : str = "p0.csv") :
    out = df.copy()
    cols_to_drop = [
        "advertisedPrice",
        "computedPrice",
        "computedPricePerSquareMeter",
        "computedSurfaceArea",
        "computedPostalCode",
        "computedBedroomsQuantity",
        "computedBathroomsQuantity",
        "computedShowerRoomsQuantity",
        "computedToiletQuantity",
        "computedPriceHasDecreased",
        "computedEnergyValue",
        "computedEnergyClassification",
        "computedMinEnergyConsumption",
        "computedMaxEnergyConsumption",
        "computedAvgEnergyConsumption",
        "computedNewProperty",
        "computedHasElevator",
        "computedHasTerrace",
    ]
    out = out.drop(columns=[c for c in cols_to_drop if c in out.columns])
    verify_required_schema(out)
    out = filter_transaction_type(df=out,ttype="rent")
    out = filter_property_type(df=out, ttype="flat")
    out = get_computedPrice(out)
    out = get_computedPricePerSquareMeter(out)
    out = get_computedSurfaceArea(out)
    out = get_computedPostalCode(out)
    out = compute_layout(out)
    out = compute_sanitary(out)
    out = get_computedPriceHasDecreased(out)
    out = compute_energy(out)
    out = get_computedEnergyConsumption(out)
    out = get_computedNewProperty(out)
    out = get_computedHasElevator(out)
    out = get_computedHasTerrace(out)
    if save :
        out.to_csv(f"ressources/out_data/{fname}")
    return out

if __name__ == "__main__" :
    DATA_PATH = Path(__file__).parent / "bienIci_2458.csv"
    df = pd.read_csv(DATA_PATH)
    out = run_pipeline(df = df)
    out.to_csv("test_cleanedDs.csv")
