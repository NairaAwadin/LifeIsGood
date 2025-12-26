OSM ENRICHMENT: WHAT IT DOES
- Goal: query a local Postgres/PostGIS database loaded with OSM data to count and find nearby POIs (restaurants, transit, parks, etc.) around a lat/lon. Results include distances and Google Maps links. The script does not download OSM data itself-it only queries a DB you provide.

PREREQUISITES (WHY/WHAT)
1) Python deps: need psycopg2-binary (for Postgres) plus the core requirements already listed in requirements.txt. Install inside your venv so the script can connect.
2) Postgres with PostGIS + hstore extensions: required because the SQL uses spatial functions (ST_DWithin, ST_Transform) and hstore tag parsing. Without PostGIS/hstore the queries fail.
3) OSM data loaded into the DB: you must load OSM planet or regional extract into the PostGIS schema (tables planet_osm_point and planet_osm_polygon). The script just reads those tables; without data you get empty results.

QUICK START (DOCKER EXAMPLE)
1) Run Postgres/PostGIS container (creates PostGIS + hstore):
   docker run -d --name osm-db -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgis/postgis:16-3.4

2) Create database + enable extensions (inside psql):
   CREATE DATABASE osm;
   \c osm
   CREATE EXTENSION postgis;
   CREATE EXTENSION hstore;

3) Load OSM extract with osm2pgsql (replace yourfile.pbf with a region extract):
   osm2pgsql -d osm -U postgres -H localhost --create --slim --hstore --latlong yourfile.pbf
   - What this does: ingests OSM PBF into planet_osm_point/polygon with hstore tags; --latlong keeps WGS84.

4) Install Python deps in your venv:
   cd "python script/analyze_bienetre"
   pip install -r requirements.txt

CONNECTION + IMPORT COMMAND SEQUENCE (ONE-TIME SETUP)
- Start container: docker start osm-db
- Create DB + extensions (once): docker exec -it osm-db psql -U postgres -c "CREATE DATABASE osm;"; docker exec -it osm-db psql -U postgres -d osm -c "CREATE EXTENSION postgis;"; docker exec -it osm-db psql -U postgres -d osm -c "CREATE EXTENSION hstore;"
- Import OSM extract (once, or when updating data): osm2pgsql -d osm -U postgres -H localhost --create --slim --hstore --latlong yourfile.pbf  (run on host with osm2pgsql installed; or run inside container after `docker exec -it osm-db bash`)
- Reuse later (daily): docker start osm-db; then run your script (python osm.py) using the same host/port/user/pass to open the connection and query the loaded data.

CONFIGURE CONNECTION (WHERE/WHY)
- In osm.py, connect_osm_db defaults to host=localhost, port=5432, dbname=osm, user=postgres, password=postgres.
- If your container/user differs, pass the right values when calling connect_osm_db or edit the defaults. Without correct creds the connection will fail.

RUNNING THE SCRIPT (WHAT TO EXPECT)
- From the project folder:
  python osm.py
- It will query predefined categories (restaurants, transit, etc.) around the sample lat/lon and print nearest POIs and counts. Adjust lat/lon/radius_m or integrate get_pois() in your pipeline to target your own coordinates.

TROUBLESHOOTING (WHY IT FAILS)
- Connection refused/auth errors: DB not running or wrong host/port/user/pass.
- "function st_dwithin does not exist": PostGIS extension missing-run CREATE EXTENSION postgis;.
- Missing hstore errors: run CREATE EXTENSION hstore;.
- Empty results: OSM tables empty or coords outside your loaded region; ensure your PBF covers the area.
