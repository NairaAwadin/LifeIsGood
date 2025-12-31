from __future__ import annotations
from typing import Any, Optional, Dict, List
#connect to postgres database, run sql, fetch results,...
import psycopg2
"""
imports on top of psychpg2 :
* RealDictCursor :
Normal fetch from cursor return tuples like, this one return dict. 
(ex : (id,name) -> {"id":1,"name":"stark"})
* register_hstroe :
Facilitate conversion between hstore type of postgres with dict in python.
"""
from psycopg2.extras import RealDictCursor, register_hstore
import time
#for constructing URL, turn " " to "+"
from urllib.parse import quote_plus

def google_maps_link(lat: float, lon: float, name: str | None = None) -> str:
    if name:
        q = quote_plus(f"{name} near {lat},{lon}")
    else:
        q = f"{lat},{lon}"
    return f"https://www.google.com/maps/search/?api=1&query={q}"

def addr_from_tags(tags: Optional[dict]) -> Optional[str]:
    """
    Docstring for addr_from_tags
    
    :param tags: Description
    :type tags: Optional[dict]
    :return: Description
    :rtype: str | None

    from tags as dict like {
    "hn": 1,
    "street" : "street bolo",
    "postcode" : "45222",
    "city" : "not-paris"
    }
    to -> parts = ['1 street bolo', '45222 not-paris']
    return -> "1 street bolo,45222 not-paris"
    """
    if not tags:
        return None
    hn = tags.get("addr:housenumber")#get value from key in a dict
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


def connect_osm_db(
    host: str = "localhost",
    port: int = 5432,
    dbname: str = "osm",
    user: str = "postgres",
    password: str = "postgres",
):
    conn = psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password)
    try:
        register_hstore(conn)  # ensures hstore columns arrive as dicts (when fetching)
    except psycopg2.ProgrammingError:
        conn.rollback()
    return conn #return live connection to postgres (ex : <connection object at 0x7f...; dsn: 'user=postgres dbname=osm host=localhost port=5432'>)


_COUNT_POINTS_SQL = """
WITH p AS (
  SELECT ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326) AS pt
)
SELECT COUNT(*)::int AS n
FROM planet_osm_point, p
WHERE ({where_clause})
  AND name IS NOT NULL
  AND ST_DWithin(way, ST_Transform(p.pt, 3857), %(r_m)s);
"""

_NEAREST_POINTS_SQL = """
WITH p AS (
  SELECT ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326) AS pt
)
SELECT
  name,
  ST_Y(ST_Transform(way, 4326)) AS lat,
  ST_X(ST_Transform(way, 4326)) AS lon,
  ST_Distance(ST_Transform(way, 4326)::geography, p.pt::geography)/1000.0 AS dist_km,
  tags
FROM planet_osm_point, p
WHERE ({where_clause})
  AND name IS NOT NULL
  AND ST_DWithin(way, ST_Transform(p.pt, 3857), %(r_m)s)
ORDER BY way <-> ST_Transform(p.pt, 3857)
{limit_clause};
"""

_NEAREST_POINT_OR_POLY_SQL = """
WITH p AS (
  SELECT ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326) AS pt
),
candidates AS (
  SELECT
    name,
  ST_Y(ST_Transform(way, 4326)) AS lat,
  ST_X(ST_Transform(way, 4326)) AS lon,
  ST_Distance(ST_Transform(way, 4326)::geography, p.pt::geography)/1000.0 AS dist_km,
  tags
FROM planet_osm_point, p
  WHERE ({where_points})
    AND name IS NOT NULL
    AND ST_DWithin(way, ST_Transform(p.pt, 3857), %(r_m)s)

  UNION ALL

  SELECT
    name,
    ST_Y(ST_Transform(ST_PointOnSurface(way), 4326)) AS lat,
    ST_X(ST_Transform(ST_PointOnSurface(way), 4326)) AS lon,
    ST_Distance(ST_Transform(ST_PointOnSurface(way), 4326)::geography, p.pt::geography)/1000.0 AS dist_km,
    tags
  FROM planet_osm_polygon, p
  WHERE ({where_polys})
    AND name IS NOT NULL
    AND ST_DWithin(ST_PointOnSurface(way), ST_Transform(p.pt, 3857), %(r_m)s)
)
SELECT *
FROM candidates
ORDER BY dist_km ASC
LIMIT 1;
"""


def enrich_location_offline(
    conn,
    lat: float,
    lon: float,
    radius_m: int = 500,
    top_n: Optional[int] = None,   # how many nearest POIs per category to return (None = all)
) -> Dict[str, Any]:
    params = {"lat": lat, "lon": lon, "r_m": radius_m}

    def count_points(where_clause: str) -> int:
        q = _COUNT_POINTS_SQL.format(where_clause=where_clause)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:#parameter indicate to return as dict
            cur.execute(q, params)#second parameter serve similar purpose as .format() but it's psycopg2/postgres that handles the formating
            return int(cur.fetchone()["n"])

    def nearest_points(where_clause: str) -> List[Dict[str, Any]]:
        limit_clause = "" if top_n is None else "LIMIT %(limit)s"
        q = _NEAREST_POINTS_SQL.format(where_clause=where_clause, limit_clause=limit_clause)
        exec_params = {**params}#same as = params.copy()
        if top_n is not None:
            exec_params["limit"] = top_n
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q, exec_params)
            rows = cur.fetchall()

        out = []
        for r in rows:
            address = addr_from_tags(r.get("tags"))
            out.append({
                "name": r.get("name"),
                "dist_km": float(r["dist_km"]) if r.get("dist_km") is not None else None,
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "address": address,
                "gmaps_url": google_maps_link(float(r["lat"]), float(r["lon"]),name=r.get("name")),
            })
        return out

    # category filters (POINTS)
    where_restaurants = "amenity IN ('restaurant','cafe','fast_food','bar','pub')"
    where_supermarkets = "shop IN ('supermarket','bakery')"
    where_bike = "amenity = 'bicycle_rental'"
    where_bus = "highway = 'bus_stop'"
    where_metro_entrance = "railway = 'subway_entrance'"
    where_rail_station = "(railway = 'station' OR public_transport = 'station')"
    where_tram = "railway = 'tram_stop'"

    categories = {
        "restaurants_like": where_restaurants,
        "supermarkets_like": where_supermarkets,
        "bike_rentals": where_bike,
        "bus_stops": where_bus,
        "metro_entrances": where_metro_entrance,
        "rail_stations": where_rail_station,
        "tram_stops": where_tram,
    }

    # counts + nearest lists for those categories
    pois = {}
    for name, where_clause in categories.items():
        pois[name] = {
            "count": count_points(where_clause),
            "nearest": nearest_points(where_clause),
        }

    # keep your “nearest” extras (park/monument/hospital)
    def nearest(where_points: str, where_polys: str) -> Optional[Dict[str, Any]]:
        q = _NEAREST_POINT_OR_POLY_SQL.format(where_points=where_points, where_polys=where_polys)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q, {"lat": lat, "lon": lon, "r_m": radius_m})
            row = cur.fetchone()

        if not row:
            return None

        address = addr_from_tags(row.get("tags"))
        return {
            "name": row.get("name"),
            "dist_km": float(row["dist_km"]) if row.get("dist_km") is not None else None,
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "address": address,
            "gmaps_url": google_maps_link(float(row["lat"]), float(row["lon"]),name=row.get("name")),
        }

    nearest_out = {
        "park": nearest("leisure IN ('park','garden')", "leisure IN ('park','garden')"),
        "monument_or_attraction": nearest(
            "(historic IN ('monument','memorial','castle') OR tourism IN ('attraction','museum','gallery'))",
            "(historic IN ('monument','memorial','castle') OR tourism IN ('attraction','museum','gallery'))",
        ),
        "hospital_or_health": nearest(
            "amenity IN ('pharmacy')",
            "amenity IN ('pharmacy')",
        ),
    }

    return {
        "input": {"lat": lat, "lon": lon, "radius_m": radius_m},
        "poi_categories": pois,     # <-- counts + nearest lists (with names + gmaps links)
        "nearest": nearest_out,     # <-- single nearest park/monument/hospital
    }


def get_pois(
    lat: float,
    lon: float,
    radius_m: int = 500
    ):
    conn = connect_osm_db(host="127.0.0.1", port=5432, dbname="osm", user="postgres", password="postgres")
    result = enrich_location_offline(conn=conn, lat = lat, lon = lon, radius_m = radius_m)
    return result

if __name__ == "__main__" :
    conn = connect_osm_db(host="127.0.0.1", port=5432, dbname="osm", user="postgres", password="postgres")
    try:
        n = 1
        elapseds = []
        for i in range(n):
            start = time.perf_counter()
            result = enrich_location_offline(conn=conn, lat = 48.8566, lon = 2.3522)
            elapseds.append(time.perf_counter() - start)
        print(result)
        for e in elapseds :
            print(f"elapsed: {e:.2f}s")
        print(f"Total elapse = {sum(elapseds):.2f}")
    finally :
        conn.close()
