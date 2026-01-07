# Mapbox Geocoding (Parallel Task)

Steps:
1. Ensure cms_data.csv exists in CMSTimerIncremental root (produced by monthly pipeline).
2. Set token (recommended):
   export MAPBOX_TOKEN="YOUR_TOKEN_HERE"
   (Or pass --token explicitly.)

3. Run dry run (no API calls):
   python mapbox_geocode.py --dry-run

4. Execute real lookups (years 2023 & 2024):
   python mapbox_geocode.py --years 2023,2024

Outputs (in Mapbox/):
- mapbox_geocode_cache.json : persistent cache
- mapbox_geocoded.csv       : CCN -> lat/lon + metadata
- cms_data_with_mapbox.csv  : dedup year subset enriched

Options:
--force         re-query even if cached / existing coords
--max-calls N   safety cap
--min-relevance adjust relevance threshold (default 0.80)
--sleep S       pacing between calls

No changes are made to original cms_data.csv.
