import os, sys, json, time, csv, argparse, urllib.parse
from pathlib import Path
import pandas as pd
import requests

MODULE_DIR = Path(__file__).resolve().parent.parent  # one level up (CMSTimerIncremental)
DEFAULT_INPUT = MODULE_DIR / "cms_data.csv"  # adjust to cms_data_visuals.csv if preferred
MAPBOX_DIR = Path(__file__).parent
CACHE_FILE = MAPBOX_DIR / "mapbox_geocode_cache.json"
OUTPUT_GEOCODED = MAPBOX_DIR / "mapbox_geocoded.csv"
OUTPUT_ENRICHED = MAPBOX_DIR / "cms_data_with_mapbox.csv"

ACCEPT_PLACE_TYPES = {"address", "poi", "place"}
DEFAULT_MIN_RELEVANCE = 0.80

def _load_cache() -> dict:
	if CACHE_FILE.exists():
		try:
			return json.loads(CACHE_FILE.read_text())
		except Exception:
			return {}
	return {}

def _save_cache(cache: dict):
	try:
		CACHE_FILE.write_text(json.dumps(cache, indent=2))
	except Exception:
		pass

def _norm_part(x):
	if x is None:
		return ""
	s = str(x).strip()
	if not s or s.lower() in {"nan","none","null"}:
		return ""
	return s

def _build_query(row) -> str:
	street = _norm_part(row.get("Address"))
	city = _norm_part(row.get("City"))
	state = _norm_part(row.get("State"))
	zip_code = _norm_part(row.get("Zip Code") or row.get("ZIP") or row.get("Zip"))
	if zip_code and len(zip_code) > 10:
		zip_code = zip_code[:10]
	parts = [p for p in [street, city, state, zip_code] if p]
	return ", ".join(parts)

def _extract_coords(feature: dict, min_relevance: float) -> tuple[float,float] | None:
	try:
		rel = feature.get("relevance", 0)
		if rel < min_relevance:
			return None
		ptypes = set(feature.get("place_type", []))
		if not ptypes & ACCEPT_PLACE_TYPES:
			return None
		coords = feature.get("center")
		if not coords or len(coords) != 2:
			return None
		lon, lat = coords
		return float(lat), float(lon)
	except Exception:
		return None

def _mapbox_lookup(query: str, token: str, min_relevance: float, timeout: int = 10) -> dict | None:
	if not query:
		return None
	url_query = urllib.parse.quote(query)
	url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{url_query}.json"
	params = {
		"access_token": token,
		"limit": 1,
		"autocomplete": "false",
		"country": "US"
	}
	try:
		resp = requests.get(url, params=params, timeout=timeout)
		if resp.status_code != 200:
			return None
		data = resp.json()
		features = data.get("features") or []
		if not features:
			return None
		feat = features[0]
		coords = _extract_coords(feat, min_relevance=min_relevance)
		if coords is None:
			return None
		return {
			"query": query,
			"lat": coords[0],
			"lon": coords[1],
			"relevance": feat.get("relevance"),
			"place_type": ";".join(feat.get("place_type", [])),
			"id": feat.get("id")
		}
	except Exception:
		return None

def _resolve_token(cli_token: str | None) -> str:
	if cli_token:
		return cli_token
	env = os.getenv("MAPBOX_TOKEN")
	if env:
		return env
	raise SystemExit("Mapbox token not supplied. Use --token or set MAPBOX_TOKEN env var.")

def _anchor_in_mapbox(p: Path) -> Path:
	"""Force any provided path into the Mapbox directory."""
	if p.is_absolute() and p.parent == MAPBOX_DIR:
		return p
	return MAPBOX_DIR / p.name

def geocode_dedup(
	input_path: Path = DEFAULT_INPUT,
	years: tuple[int,int] = (2023, 2024),
	token: str | None = None,
	force: bool = False,
	max_calls: int | None = None,
	sleep_sec: float = 0.12,
	min_relevance: float = DEFAULT_MIN_RELEVANCE,
	dry_run: bool = False,
	fresh: bool = False,
	output_path: Path | None = None,
	verbose: bool = False,
	progress_every: int = 50,
) -> dict:
	# Resolve output path (anchor into Mapbox folder)
	output_file = _anchor_in_mapbox(Path(output_path)) if output_path else OUTPUT_GEOCODED

	# Fresh start handling
	if fresh:
		if CACHE_FILE.exists():
			try: CACHE_FILE.unlink()
			except Exception: pass
		for f in (output_file, OUTPUT_ENRICHED):
			if f.exists():
				try: f.unlink()
				except Exception: pass
		# force full re-query
		force = True

	token_val = _resolve_token(token)
	if not input_path.exists():
		raise FileNotFoundError(f"Input file not found: {input_path}")
	df = pd.read_csv(input_path, low_memory=False)
	if "Year" not in df.columns:
		raise ValueError("Input file missing 'Year' column.")
	if "CCN" not in df.columns:
		raise ValueError("Input file missing 'CCN' column.")
	target = df[df["Year"].isin(years)].copy()
	target = target.sort_values(["CCN","Year"]).drop_duplicates(subset=["CCN"], keep="last")
	total_candidates = len(target)

	has_latlon = {"Latitude","Longitude"}.issubset(target.columns)
	if not has_latlon:
		target["Latitude"] = pd.NA
		target["Longitude"] = pd.NA

	# If fresh, blank out to guarantee new calls even if previously geocoded
	if fresh:
		target["Latitude"] = pd.NA
		target["Longitude"] = pd.NA

	if force:
		to_geo = target
	else:
		to_geo = target[target["Latitude"].isna() | target["Longitude"].isna()]
	skip_count = total_candidates - len(to_geo)

	if total_candidates > 80000:
		print(f"[WARN] Large geocode set ({total_candidates}). Verify this is intended before proceeding.")

	# Collect which address columns exist
	address_cols = [c for c in ["Address","City","State","Zip Code"] if c in target.columns]

	cache = _load_cache()
	results = []
	call_count = 0
	success = 0
	fail = 0

	start_ts = time.time()
	if verbose:
		print(f"[START] total_candidates={total_candidates} to_lookup_initial={(len(target) if force else (target['Latitude'].isna() | target['Longitude'].isna()).sum())} fresh={fresh} force={force}", flush=True)
	else:
		print(f"[START] Geocoding {len(to_geo)} of {total_candidates} candidates (fresh={fresh}, force={force})", flush=True)

	def _print_progress(call_count: int, success: int, fail: int, total: int, start_ts: float, verbose: bool):
		elapsed = time.time() - start_ts
		percent = (call_count / total * 100) if total else 0
		rate = call_count / elapsed if elapsed > 0 else 0
		remain = total - call_count
		eta = remain / rate if rate > 0 else 0
		if verbose:
			print(f"[PROGRESS] calls={call_count} success={success} fail={fail} rate={rate:.1f}/s pct={percent:.1f}% eta={eta/60:.1f}m", flush=True)
		else:
			print(f"[PROGRESS] {call_count}/{total} ({percent:.1f}%) success={success} fail={fail} ETA={int(eta//60):02d}:{int(eta%60):02d}", flush=True)

	for idx, row in to_geo.iterrows():
		ccn = str(row["CCN"])
		query = _build_query(row)
		if not query:
			fail += 1
			continue
		cache_key = f"CCN::{ccn}"
		# Only reuse cache if not force
		if cache_key in cache and not force:
			entry = cache[cache_key]
			rec = {
				"CCN": ccn,
				"Latitude": entry["lat"],
				"Longitude": entry["lon"],
				"relevance": entry.get("relevance"),
				"place_type": entry.get("place_type"),
				"source": "cache",
				"query": entry.get("query","")
			}
			for col in address_cols:
				rec[col] = row.get(col)
			results.append(rec)
			continue
		if max_calls is not None and call_count >= max_calls:
			break
		if dry_run:
			call_count += 1
			continue
		lookup = _mapbox_lookup(query, token_val, min_relevance=min_relevance)
		call_count += 1
		if lookup:
			cache[cache_key] = {
				"query": lookup["query"],
				"lat": lookup["lat"],
				"lon": lookup["lon"],
				"relevance": lookup["relevance"],
				"place_type": lookup["place_type"],
				"id": lookup["id"]
			}
			rec = {
				"CCN": ccn,
				"Latitude": lookup["lat"],
				"Longitude": lookup["lon"],
				"relevance": lookup["relevance"],
				"place_type": lookup["place_type"],
				"source": "mapbox",
				"query": lookup["query"]
			}
			for col in address_cols:
				rec[col] = row.get(col)
			results.append(rec)
			success += 1
		else:
			fail += 1
		time.sleep(sleep_sec)
		if call_count and (call_count % progress_every == 0):
			_print_progress(call_count, success, fail, len(to_geo), start_ts, verbose)

	# Final progress (if not already exact multiple)
	if call_count % progress_every != 0:
		_print_progress(call_count, success, fail, len(to_geo), start_ts, verbose)

	if not dry_run and results:
		out_df = pd.DataFrame(results)

		# Ensure all expected columns present
		expected_cols = ["CCN","Address","City","State","Zip Code","Latitude","Longitude","relevance","place_type","source","query"]
		for c in expected_cols:
			if c not in out_df.columns:
				out_df[c] = pd.NA

		# Merge with prior (if not fresh)
		if output_file.exists() and not fresh:
			try:
				prev = pd.read_csv(output_file)
				# Add any missing columns
				for c in expected_cols:
					if c not in prev.columns:
						prev[c] = pd.NA
				combined = pd.concat([prev[expected_cols], out_df[expected_cols]], ignore_index=True)
				# Prefer mapbox over cache; keep last occurrence if same source
				combined["__priority"] = combined["source"].map({"mapbox": 2, "cache": 1}).fillna(0)
				combined = (combined
							.sort_values(["CCN","__priority"])
							.drop_duplicates(subset=["CCN"], keep="last")
							.drop(columns="__priority"))
				out_df = combined
			except Exception:
				pass

		out_df.to_csv(output_file, index=False)

		# Enriched (unchanged logic, but reuse new output with addresses)
		enriched = target.merge(out_df[["CCN","Latitude","Longitude"]], on="CCN", how="left", suffixes=("","_mb"))
		mask_fill = (enriched["Latitude"].isna() | enriched["Longitude"].isna()) & enriched["Latitude_mb"].notna()
		enriched.loc[mask_fill, "Latitude"] = enriched.loc[mask_fill, "Latitude_mb"]
		enriched.loc[mask_fill, "Longitude"] = enriched.loc[mask_fill, "Longitude_mb"]
		enriched.drop(columns=["Latitude_mb","Longitude_mb"], errors="ignore", inplace=True)
		enriched.to_csv(OUTPUT_ENRICHED, index=False)
		_save_cache(cache)

	if verbose:
		elapsed = time.time() - start_ts
		print(f"[DONE] calls={call_count} success={success} fail={fail} elapsed={elapsed:.1f}s", flush=True)
	else:
		print(f"[DONE] calls={call_count} success={success} fail={fail}", flush=True)
	return {
		"fresh": fresh,
		"output_file": str(output_file),
		"cache_cleared": fresh,
		"outputs_deleted": fresh,
		"total_candidates": total_candidates,
		"already_geocoded_or_skipped": skip_count,
		"to_lookup": len(to_geo),
		"api_calls_made": call_count,
		"success_new": success,
		"failed": fail,
		"results_written": output_file.exists() and not dry_run,
		"enriched_written": OUTPUT_ENRICHED.exists() and not dry_run,
		"cache_size": len(cache),
		"elapsed_sec": round(time.time() - start_ts, 2)
	}

def parse_args(argv=None):
	p = argparse.ArgumentParser(description="Mapbox geocoding for CMS hospitals (parallel task).")
	p.add_argument("--input", default=str(DEFAULT_INPUT), help="Input CSV (cms_data.csv or cms_data_visuals.csv).")
	p.add_argument("--years", default="2023,2024", help="Comma list of years (e.g. 2023,2024).")
	p.add_argument("--token", default=None, help="Mapbox token (or set MAPBOX_TOKEN env).")
	p.add_argument("--force", action="store_true", help="Force re-query even if cached / existing lat-lon.")
	p.add_argument("--max-calls", type=int, default=None, help="Cap API calls (safety).")
	p.add_argument("--sleep", type=float, default=0.12, help="Sleep seconds between calls.")
	p.add_argument("--min-relevance", type=float, default=DEFAULT_MIN_RELEVANCE, help="Minimum relevance accepted.")
	p.add_argument("--dry-run", action="store_true", help="Do not perform API calls; show summary only.")
	p.add_argument("--fresh", action="store_true", help="Delete cache & prior outputs; re-query all candidates.")
	p.add_argument("--output", default=None, help="Override geocode results CSV path (anchored in Mapbox folder).")
	p.add_argument("--verbose", action="store_true", help="Verbose progress output with rates.")
	p.add_argument("--progress-every", type=int, default=50, help="Print progress every N API calls.")
	return p.parse_args(argv)

def main(argv=None):
	args = parse_args(argv)
	years = tuple(int(y.strip()) for y in args.years.split(",") if y.strip())
	info = geocode_dedup(
		input_path=Path(args.input),
		years=years,
		token=args.token,
		force=args.force,
		max_calls=args.max_calls,
		sleep_sec=args.sleep,
		min_relevance=args.min_relevance,
		dry_run=args.dry_run,
		fresh=args.fresh,
		output_path=Path(args.output) if args.output else None,
		verbose=args.verbose,
		progress_every=args.progress_every,
	)
	print(json.dumps(info, indent=2))

if __name__ == "__main__":
	main()
