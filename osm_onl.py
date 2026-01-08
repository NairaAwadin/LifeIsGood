from __future__ import annotations

from typing import Any, Optional, Dict, List, Tuple
from urllib.parse import quote_plus
import math
import time
import requests


def google_maps_link(lat: float, lon: float, name: str | None = None) -> str:
    if name:
        q = quote_plus(f"{name} near {lat},{lon}")
    else:
        q = f"{lat},{lon}"
    return f"https://www.google.com/maps/search/?api=1&query={q}"


def addr_from_tags(tags: Optional[dict]) -> Optional[str]:
    if not tags:
        return None
    hn = tags.get("addr:housenumber")
    street = tags.get("addr:street")
    postcode = tags.get("addr:postcode")
    city = tags.get("addr:city")

    parts = []
    if street:
        parts.append(f"{(hn + ' ') if hn else ''}{street}".strip())
    pc_city = " ".join([p for p in [postcode, city] if p])
    if pc_city:
        parts.append(pc_city)
    return ", ".join(parts) if parts else None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Earth radius (km)
    R = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


#Overpass client
DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"  # main instance :contentReference[oaicite:1]{index=1}

HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "accept": "application/json",
}


def overpass_fetch_elements(
    query: str,
    base_url: str = DEFAULT_OVERPASS_URL,
    timeout_s: int = 60,
    max_retries: int = 3,
    backoff_s: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    Executes an Overpass QL query and returns the `elements` list.
    Uses POST (recommended for longer queries).
    Retries on transient errors (429/502/504).
    """
    last_err: Exception | None = None

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                base_url,
                headers=HEADERS,
                data={"data": query},
                timeout=timeout_s,
            )

            # common transient statuses
            if resp.status_code in (429, 502, 504):
                time.sleep(backoff_s * (2 ** attempt))
                continue

            resp.raise_for_status()
            payload = resp.json()

            # Overpass sometimes returns an "remark" on error
            if isinstance(payload, dict) and payload.get("remark"):
                raise RuntimeError(f"Overpass remark: {payload['remark']}")

            elements = payload.get("elements", [])
            if not isinstance(elements, list):
                return []
            return elements

        except Exception as e:
            last_err = e
            time.sleep(backoff_s * (2 ** attempt))

    raise RuntimeError(f"Overpass request failed after {max_retries} retries: {last_err}")


def build_around_query(
    lat: float,
    lon: float,
    radius_m: int,
    tag_filter: str,
    timeout_s: int = 25,
) -> str:
    """
    tag_filter example: '["amenity"~"restaurant|cafe"]' or '["shop"="supermarket"]'
    Uses (around:radius,lat,lon) syntax. :contentReference[oaicite:2]{index=2}
    """
    # We query node + way + relation, and ask for center for non-node elements.
    return f"""
[out:json][timeout:{int(timeout_s)}];
(
  node{tag_filter}(around:{int(radius_m)},{lat},{lon});
  way{tag_filter}(around:{int(radius_m)},{lat},{lon});
  relation{tag_filter}(around:{int(radius_m)},{lat},{lon});
);
out tags center;
""".strip()


def element_coord(el: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    t = el.get("type")
    if t == "node":
        if "lat" in el and "lon" in el:
            return float(el["lat"]), float(el["lon"])
        return None
    # way / relation: we requested "out center"
    c = el.get("center")
    if isinstance(c, dict) and "lat" in c and "lon" in c:
        return float(c["lat"]), float(c["lon"])
    return None


def element_name_tags(el: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    tags = el.get("tags") or {}
    if not isinstance(tags, dict):
        tags = {}
    name = tags.get("name")
    return name, tags


def elements_to_nearest_list(
    elements: List[Dict[str, Any]],
    lat: float,
    lon: float,
    top_n: Optional[int],
) -> List[Dict[str, Any]]:
    rows = []
    seen = set()

    for el in elements:
        el_type = el.get("type")
        el_id = el.get("id")
        if el_type is None or el_id is None:
            continue
        k = (el_type, int(el_id))
        if k in seen:
            continue
        seen.add(k)

        coord = element_coord(el)
        if coord is None:
            continue
        el_lat, el_lon = coord

        name, tags = element_name_tags(el)
        # keep unnamed POIs out (like your offline version)
        if not name:
            continue

        dist_km = haversine_km(lat, lon, el_lat, el_lon)
        address = addr_from_tags(tags)

        rows.append(
            {
                "name": name,
                "dist_km": float(dist_km),
                "lat": float(el_lat),
                "lon": float(el_lon),
                "address": address,
                "gmaps_url": google_maps_link(float(el_lat), float(el_lon), name=name),
            }
        )

    rows.sort(key=lambda r: r["dist_km"])
    if top_n is not None:
        return rows[: int(top_n)]
    return rows


def elements_count_named(elements: List[Dict[str, Any]]) -> int:
    seen = set()
    n = 0
    for el in elements:
        el_type = el.get("type")
        el_id = el.get("id")
        if el_type is None or el_id is None:
            continue
        k = (el_type, int(el_id))
        if k in seen:
            continue
        seen.add(k)

        name, _tags = element_name_tags(el)
        if name:
            n += 1
    return n


def enrich_location_online(
    lat: float,
    lon: float,
    radius_m: int = 500,
    top_n: Optional[int] = None,
    base_url: str = DEFAULT_OVERPASS_URL,
) -> Dict[str, Any]:
    # Category filters
    # Note: using regex for multi-values is common in Overpass QL. :contentReference[oaicite:3]{index=3}
    categories: Dict[str, str] = {
        "restaurants_like": '["amenity"~"restaurant|cafe|fast_food|bar|pub"]',
        "supermarkets_like": '["shop"~"supermarket|bakery"]',
        "bike_rentals": '["amenity"="bicycle_rental"]',
        "bus_stops": '["highway"="bus_stop"]',
        "metro_entrances": '["railway"="subway_entrance"]',
        "rail_stations": '["railway"="station"]',
        "tram_stops": '["railway"="tram_stop"]',
    }

    pois: Dict[str, Any] = {}
    for cat_name, tag_filter in categories.items():
        q = build_around_query(lat=lat, lon=lon, radius_m=radius_m, tag_filter=tag_filter)
        elements = overpass_fetch_elements(q, base_url=base_url)
        pois[cat_name] = {
            "count": elements_count_named(elements),
            "nearest": elements_to_nearest_list(elements, lat=lat, lon=lon, top_n=top_n),
        }

    # “nearest extras” (single nearest)
    def nearest_one(tag_filter: str) -> Optional[Dict[str, Any]]:
        q = build_around_query(lat=lat, lon=lon, radius_m=radius_m, tag_filter=tag_filter)
        elements = overpass_fetch_elements(q, base_url=base_url)
        nearest_list = elements_to_nearest_list(elements, lat=lat, lon=lon, top_n=1)
        return nearest_list[0] if nearest_list else None

    nearest_out = {
        "park": nearest_one('["leisure"~"park|garden"]'),
        "monument_or_attraction": nearest_one(
            '["historic"~"monument|memorial|castle"]'
        ) or nearest_one(
            '["tourism"~"attraction|museum|gallery"]'
        ),
        "hospital_or_health": nearest_one('["amenity"="pharmacy"]'),
    }

    return {
        "input": {"lat": lat, "lon": lon, "radius_m": radius_m},
        "poi_categories": pois,
        "nearest": nearest_out,
        "meta": {"provider": "overpass", "endpoint": base_url},
    }


def get_pois(
    lat: float,
    lon: float,
    radius_m: int = 500,
    top_n: Optional[int] = None,
    base_url: str = DEFAULT_OVERPASS_URL,
) -> Dict[str, Any]:
    return enrich_location_online(
        lat=lat,
        lon=lon,
        radius_m=radius_m,
        top_n=top_n,
        base_url=base_url,
    )


if __name__ == "__main__":
    # quick test (Paris center)
    out = get_pois(lat=48.8566, lon=2.3522, radius_m=500, top_n=5)
    print(out)
