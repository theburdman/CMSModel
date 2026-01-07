import os, json, time, argparse, urllib.parse
from pathlib import Path
import pandas as pd
import requests

MODULE_DIR = Path(__file__).resolve().parent.parent
GOOGLE_DIR = Path(__file__).parent

DEFAULT_INPUT = MODULE_DIR / "cms_data.csv"
CACHE_FILE = GOOGLE_DIR / "google_geocode_cache.json"
OUTPUT_GEOCODED = GOOGLE_DIR / "google_geocoded.csv"
OUTPUT_ENRICHED = GOOGLE_DIR / "cms_data_with_google.csv"

ACCEPT_STATUSES = {"OK"}
DEFAULT_MIN_CONFIDENCE = 0.0  # Placeholder if future confidence filtering added.

def _load_cache():
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

def _resolve_key(cli_key: str | None):
	if cli_key:
		return cli_key
	env = os.getenv("GOOGLE_API_KEY")
	if env:
		return env
	raise SystemExit("Google API key required. Use --key or set GOOGLE_API_KEY env var.")

def _google_lookup(query: str, key: str, timeout: int = 8) -> dict | None:
	if not query:
		return None
	base = "https://maps.googleapis.com/maps/api/geocode/json"
	params = {"address": query, "key": key}
	try:
		resp = requests.get(base, params=params, timeout=timeout)
		if resp.status_code != 200:
			return None
		data = resp.json()
		status = data.get("status")
		if status not in ACCEPT_STATUSES:
			return {"status": status, "error_message": data.get("error_message")}
		results = data.get("results") or []
		if not results:
			return {"status": status}
		first = results[0]
		loc = (first.get("geometry", {}).get("location") or {})
		lat, lng = loc.get("lat"), loc.get("lng")
		if lat is None or lng is None:
			return {"status": status}
		return {
			"status": status,
			"lat": float(lat),
			"lon": float(lng),
			"place_id": first.get("place_id"),
			"types": ";".join(first.get("types", [])),
			"formatted_address": first.get("formatted_address")
		}
	except Exception:
		return None

def _norm_ccn(val) -> str:
	"""Return stable CCN string (remove trailing .0 for floats, strip whitespace)."""
	if val is None:
		return ""
	if isinstance(val, float):
		if val.is_integer():
			return str(int(val))
	return str(val).strip()

def geocode_google(
	input_path: Path = DEFAULT_INPUT,
	years: tuple[int,int] = (2023, 2024),
	key: str | None = None,
	force: bool = False,
	max_calls: int | None = None,
	sleep_sec: float = 0.10,
	dry_run: bool = False,
	fresh: bool = False,
	verbose: bool = False,
	progress_every: int = 50,
) -> dict:
	out_file = OUTPUT_GEOCODED
	enriched_file = OUTPUT_ENRICHED
	if fresh:
		for f in (CACHE_FILE, out_file, enriched_file):
			if f.exists():
				try: f.unlink()
				except Exception: pass
		force = True

	api_key = _resolve_key(key)
	if not input_path.exists():
		raise FileNotFoundError(f"Input not found: {input_path}")
	df = pd.read_csv(input_path, low_memory=False)
	if "Year" not in df.columns or "CCN" not in df.columns:
		raise ValueError("Input must contain 'Year' and 'CCN' columns.")
	df["CCN"] = df["CCN"].apply(_norm_ccn)
	target = df[df["Year"].isin(years)].copy()
	target["CCN"] = target["CCN"].apply(_norm_ccn)
	target = target.sort_values(["CCN","Year"]).drop_duplicates(subset=["CCN"], keep="last")
	total = len(target)

	if {"Latitude","Longitude"}.issubset(target.columns):
		pass
	else:
		target["Latitude"] = pd.NA
		target["Longitude"] = pd.NA

	if fresh:
		target["Latitude"] = pd.NA
		target["Longitude"] = pd.NA

	to_geo = target if force else target[target["Latitude"].isna() | target["Longitude"].isna()]
	skip_count = total - len(to_geo)

	address_cols = [c for c in ["Address","City","State","Zip Code"] if c in target.columns]

	cache = _load_cache()
	results = []
	call_count = success = fail = 0
	start_ts = time.time()

	if verbose:
		print(f"[START] candidates={total} lookup={len(to_geo)} fresh={fresh} force={force}", flush=True)
	else:
		print(f"[START] Geocoding {len(to_geo)} of {total} (fresh={fresh}, force={force})", flush=True)

	def _progress():
		elapsed = time.time() - start_ts
		percent = (call_count / len(to_geo) * 100) if len(to_geo) else 0
		rate = call_count / elapsed if elapsed > 0 else 0
		remain = len(to_geo) - call_count
		eta = remain / rate if rate > 0 else 0
		if verbose:
			print(f"[PROGRESS] calls={call_count} success={success} fail={fail} pct={percent:.1f}% rate={rate:.1f}/s eta={eta/60:.1f}m", flush=True)
		else:
			m = int(eta // 60); s = int(eta % 60)
			print(f"[PROGRESS] {call_count}/{len(to_geo)} ({percent:.1f}%) success={success} fail={fail} ETA={m:02d}:{s:02d}", flush=True)

	for idx, row in to_geo.iterrows():
		ccn = _norm_ccn(row["CCN"])
		query = _build_query(row)
		if not query:
			fail += 1
			continue
		cache_key = f"CCN::{ccn}"
		if cache_key in cache and not force:
			entry = cache[cache_key]
			rec = {
				"CCN": ccn,
				"Latitude": entry.get("lat"),
				"Longitude": entry.get("lon"),
				"status": entry.get("status"),
				"source": "cache",
				"query": entry.get("query",""),
				"types": entry.get("types",""),
				"place_id": entry.get("place_id"),
				"formatted_address": entry.get("formatted_address")
			}
			for c in address_cols:
				rec[c] = row.get(c)
			results.append(rec)
			continue
		if max_calls is not None and call_count >= max_calls:
			break
		if dry_run:
			call_count += 1
			continue
		lookup = _google_lookup(query, api_key)
		call_count += 1
		if lookup and lookup.get("lat") is not None:
			cache[cache_key] = {
				"query": query,
				"lat": lookup["lat"],
				"lon": lookup["lon"],
				"status": lookup.get("status"),
				"types": lookup.get("types"),
				"place_id": lookup.get("place_id"),
				"formatted_address": lookup.get("formatted_address")
			}
			rec = {
				"CCN": ccn,
				"Latitude": lookup["lat"],
				"Longitude": lookup["lon"],
				"status": lookup.get("status"),
				"source": "google",
				"query": query,
				"types": lookup.get("types"),
				"place_id": lookup.get("place_id"),
				"formatted_address": lookup.get("formatted_address")
			}
			for c in address_cols:
				rec[c] = row.get(c)
			results.append(rec)
			success += 1
		else:
			rec_fail = {
				"CCN": ccn,
				"Latitude": pd.NA,
				"Longitude": pd.NA,
				"status": lookup.get("status") if isinstance(lookup, dict) else "ERROR",
				"source": "google" if lookup else "error",
				"query": query
			}
			for c in address_cols:
				rec_fail[c] = row.get(c)
			results.append(rec_fail)
			fail += 1
		time.sleep(sleep_sec)
		if call_count and (call_count % progress_every == 0):
			_progress()

	if call_count % progress_every != 0:
		_progress()

	if not dry_run and results:
		out_df = pd.DataFrame(results)
		out_df["CCN"] = out_df["CCN"].apply(_norm_ccn)
		expected_cols = ["CCN","Address","City","State","Zip Code","Latitude","Longitude","status","source","query","types","place_id","formatted_address"]
		for c in expected_cols:
			if c not in out_df.columns:
				out_df[c] = pd.NA
		if out_file.exists() and not fresh:
			try:
				prev = pd.read_csv(out_file)
				for c in expected_cols:
					if c not in prev.columns:
						prev[c] = pd.NA
				combined = pd.concat([prev[expected_cols], out_df[expected_cols]], ignore_index=True)
				combined["__priority"] = combined["source"].map({"google": 2, "cache": 1}).fillna(0)
				combined = (combined
							.sort_values(["CCN","__priority"])
							.drop_duplicates(subset=["CCN"], keep="last")
							.drop(columns="__priority"))
				out_df = combined
			except Exception:
				pass
		out_df.to_csv(out_file, index=False)

		enriched = target.merge(out_df[["CCN","Latitude","Longitude"]], on="CCN", how="left", suffixes=("","_g"))
		mask_fill = (enriched["Latitude"].isna() | enriched["Longitude"].isna()) & enriched["Latitude_g"].notna()
		enriched.loc[mask_fill, "Latitude"] = enriched.loc[mask_fill, "Latitude_g"]
		enriched.loc[mask_fill, "Longitude"] = enriched.loc[mask_fill, "Longitude_g"]
		enriched.drop(columns=["Latitude_g","Longitude_g"], errors="ignore", inplace=True)
		enriched.to_csv(enriched_file, index=False)
		_save_cache(cache)

	elapsed = time.time() - start_ts
	if verbose:
		print(f"[DONE] calls={call_count} success={success} fail={fail} elapsed={elapsed:.1f}s", flush=True)
	else:
		print(f"[DONE] calls={call_count} success={success} fail={fail}", flush=True)

	return {
		"fresh": fresh,
		"output_file": str(out_file),
		"enriched_file": str(enriched_file),
		"total_candidates": total,
		"skipped_existing": skip_count,
		"lookups_attempted": len(to_geo),
		"api_calls": call_count,
		"success": success,
		"fail": fail,
		"cache_size": len(cache),
		"elapsed_sec": round(elapsed, 2)
	}

def parse_args(argv=None):
	p = argparse.ArgumentParser(description="Google Geocoding (parallel task, no CMS code modifications).")
	p.add_argument("--input", default=str(DEFAULT_INPUT), help="Input CSV (cms_data.csv or cms_data_visuals.csv).")
	p.add_argument("--years", default="2023,2024", help="Comma list of years.")
	p.add_argument("--key", default=None, help="Google API key (or set GOOGLE_API_KEY env).")
	p.add_argument("--force", action="store_true", help="Re-query even if cached or lat/lon present.")
	p.add_argument("--fresh", action="store_true", help="Clear cache & outputs; full rebuild.")
	p.add_argument("--max-calls", type=int, default=None, help="Limit API calls (safety).")
	p.add_argument("--sleep", type=float, default=0.10, help="Sleep seconds between calls.")
	p.add_argument("--dry-run", action="store_true", help="No API calls; summary only.")
	p.add_argument("--verbose", action="store_true", help="Verbose progress (rate/ETA).")
	p.add_argument("--progress-every", type=int, default=50, help="Progress print frequency.")
	return p.parse_args(argv)

def main(argv=None):
	args = parse_args(argv)
	years = tuple(int(y.strip()) for y in args.years.split(",") if y.strip())
	info = geocode_google(
		input_path=Path(args.input),
		years=years,
		key=args.key,
		force=args.force,
		max_calls=args.max_calls,
		sleep_sec=args.sleep,
		dry_run=args.dry_run,
		fresh=args.fresh,
		verbose=args.verbose,
		progress_every=args.progress_every,
	)
	print(json.dumps(info, indent=2))

if __name__ == "__main__":
	main()
