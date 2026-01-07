from __future__ import annotations
import datetime
import numpy as np
import pandas as pd
import re, json, time, requests, zipfile
from io import BytesIO
from pathlib import Path
from azure.storage.blob import BlobServiceClient
import os, logging, requests, zipfile
from io import BytesIO
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import uuid

# Ensure logger early
logger = logging.getLogger(__name__)

try:
	from .ingestion import assemble_year_from_files
except ImportError:
	try:
		from ingestion import assemble_year_from_files  # type: ignore
	except ImportError:
		assemble_year_from_files = None

try:
	from .metrics import assemble_master_incremental
except ImportError:
	try:
		from metrics import assemble_master_incremental  # type: ignore
	except ImportError:
		assemble_master_incremental = None

try:
	from .model_debt import train_debt_model  # package context
except ImportError:
	try:
		from model_debt import train_debt_model  # loose script context
	except ImportError:
		train_debt_model = None

# ---------- Constants (define BEFORE any function uses them) ----------
_REQUIRED_MODEL_COLS = [
	"Net Income","EBITDAR","Total Current Liabilities","Total Long-Term Liabilities","Cash",
	"Operating Expense","Other Expense","Government Reimbursements","Net Receivables",
	"Total Current Assets","Total Debt","Net Income Average",
	">$400k EBITDAR","Debt <$5M or 5x NI","Positive Cash Position","Cash vs. 65 Days",
	"Income vs. 3% Total Expenses","Assets vs. 100 Days","$100k Reimbursement",
	"Net Receivables vs. 88 Days","Rural vs. Urban","Number Beds","Type of Control"
]

MODULE_DIR = Path(__file__).parent.resolve()

# Google geocode file constants
_GOOGLE_GEOCODE_CSV = MODULE_DIR / "google_geocoded.csv"
_GOOGLE_USAGE_DIR = MODULE_DIR
_GOOGLE_MONTHLY_LIMIT = 10_000
GOOGLE_GEOCODE_BLOB_NAME = "google_geocoded.csv"

BASE_BLOB_NAME = "cms_data.csv"
VISUALS_BLOB_NAME = "cms_data_visuals.csv"

def _module_path(name: str | Path) -> Path:
	p = Path(name)
	return MODULE_DIR / p.name if not p.is_absolute() else p

# Replace cache file path to reside in module dir
_CACHE_FILE = _module_path("geocode_cache.json")

def _prior_calendar_year():
    return datetime.datetime.now().year - 1

def _filter_base_columns(df: pd.DataFrame) -> pd.DataFrame:
	"""Do not truncate columns; return as-is."""
	return df

def _clean_zip(z):
	"""Return valid 5-digit ZIP or empty (reject '00000', short, malformed)."""
	if z is None or (isinstance(z, float) and pd.isna(z)):
		return ""
	s = str(z).strip()
	m = re.findall(r'\d{5}', s)
	if not m:
		return ""
	zip5 = m[0]
	return "" if zip5 == "00000" else zip5

def _strip_visuals_columns(df: pd.DataFrame) -> pd.DataFrame:
	"""Remove columns that belong only in visuals output."""
	for col in ["Latitude","Longitude","Potential_CAH_Google","Min_Distance_Miles","Full Address"]:
		if col in df.columns:
			df = df.drop(columns=[col])
	return df

def _dedupe_core(df: pd.DataFrame) -> pd.DataFrame:
	"""Drop duplicate rows on (Report Record Number, Year) keeping first."""
	if "Report Record Number" in df.columns and "Year" in df.columns:
		return df.drop_duplicates(subset=["Report Record Number","Year"], keep="first")
	return df

def _select_recent_years(df: pd.DataFrame) -> list[int]:
	if "Year" not in df.columns:
		return []
	years = pd.to_numeric(df["Year"], errors="coerce").dropna().astype(int).unique()
	if years.size == 0:
		return []
	return sorted(years, reverse=True)[:2]

def _norm_ccn(v) -> str:
	# ensure consistent CCN normalization everywhere
	if v is None or (isinstance(v,float) and pd.isna(v)):
		return ""
	s = str(v).strip()
	return s[:-2] if s.endswith('.0') else s

def _norm_address_parts(street, city, state, zip_code) -> str:
	def _p(x):
		if x is None or (isinstance(x, float) and pd.isna(x)):
			return ""
		s = str(x).strip()
		if not s or s.lower() in {"nan","none","null"}:
			return ""
		return s
	return ", ".join([_p(street), _p(city), _p(state), _p(str(zip_code)[:10])]).strip(", ")

def _google_usage_file(dt: datetime):
	return _GOOGLE_USAGE_DIR / f"google_usage_{dt.strftime('%Y%m')}.json"

def _load_month_usage(dt: datetime) -> dict:
	f = _google_usage_file(dt)
	if f.exists():
		try:
			return json.loads(f.read_text())
		except Exception:
			return {"count": 0}
	return {"count": 0}

def _save_month_usage(dt: datetime, usage: dict):
	try:
		_google_usage_file(dt).write_text(json.dumps(usage))
	except Exception:
		pass

def _load_existing_google(cache_csv: Path = _GOOGLE_GEOCODE_CSV) -> pd.DataFrame:
	if cache_csv.exists():
		try:
			df = pd.read_csv(cache_csv)
			df["CCN"] = df["CCN"].apply(_norm_ccn)
			return df
		except Exception:
			return pd.DataFrame(columns=["CCN","Latitude","Longitude","Address","City","State","Zip Code"])
	return pd.DataFrame(columns=["CCN","Latitude","Longitude","Address","City","State","Zip Code"])

def _google_lookup(query: str, api_key: str, timeout: int = 8) -> tuple[float,float] | None:
	if not query:
		return None
	url = "https://maps.googleapis.com/maps/api/geocode/json"
	params = {"address": query, "key": api_key}
	try:
		r = requests.get(url, params=params, timeout=timeout)
		if r.status_code != 200:
			return None
		data = r.json()
		if data.get("status") != "OK":
			return None
		results = data.get("results") or []
		if not results:
			return None
		loc = results[0].get("geometry", {}).get("location", {})
		lat, lng = loc.get("lat"), loc.get("lng")
		if lat is None or lng is None:
			return None
		return float(lat), float(lng)
	except Exception:
		return None

def _apply_geocodes_from_cache(df: pd.DataFrame) -> pd.DataFrame:
	"""Map lat/long from google_geocoded.csv onto df by CCN without dropping existing valid coords."""
	cache = _load_geocode_cache()
	if cache.empty or "CCN" not in df.columns:
		return df
	cache = cache.dropna(subset=["Latitude","Longitude"])
	if cache.empty:
		return df
	cache_map = cache.set_index("CCN")[["Latitude","Longitude"]].to_dict("index")
	def _map(r):
		ccn = str(r["CCN"]).strip()
		if ccn in cache_map and (pd.isna(r.get("Latitude")) or pd.isna(r.get("Longitude"))):
			return cache_map[ccn]["Latitude"], cache_map[ccn]["Longitude"]
		return r.get("Latitude"), r.get("Longitude")
	df["Latitude"], df["Longitude"] = zip(*df.apply(_map, axis=1))
	return df

def attach_google_geocodes(base_df: pd.DataFrame,
						   years: list[int],
						   force: bool = False) -> pd.DataFrame:
	if "CCN" not in base_df.columns or "Year" not in base_df.columns:
		return base_df
	if "Latitude" not in base_df.columns:
		base_df["Latitude"] = np.nan
	if "Longitude" not in base_df.columns:
		base_df["Longitude"] = np.nan
	# Use cached geocodes first
	base_df = _apply_geocodes_from_cache(base_df)
	sub = base_df[base_df["Year"].isin(years)].copy()
	sub["CCN"] = sub["CCN"].apply(_norm_ccn)
	for col in ["Address","City","State","Zip Code"]:
		if col not in sub.columns:
			sub[col] = ""
	sub["__addr_key"] = sub.apply(lambda r: _norm_address_parts(r["Address"], r["City"], r["State"], r["Zip Code"]), axis=1)
	sub = sub.sort_values(["CCN","Year"]).drop_duplicates(subset=["CCN"], keep="last")

	cache_df = _load_existing_google()
	cache_df["__addr_key"] = cache_df.apply(lambda r: _norm_address_parts(r.get("Address"), r.get("City"), r.get("State"), r.get("Zip Code")), axis=1)

	merged = sub.merge(cache_df[["CCN","Latitude","Longitude","__addr_key"]], on="CCN", how="left", suffixes=("","_cached"))
	api_key = os.getenv("GOOGLE_API_KEY")
	month_usage = _load_month_usage(datetime.utcnow())
	remaining = max(0, _GOOGLE_MONTHLY_LIMIT - month_usage.get("count", 0))
	do_api = api_key is not None and remaining > 0

	to_query_mask = (
		(force |
		 merged["Latitude_cached"].isna() |
		 merged["Longitude_cached"].isna() |
		 (merged["__addr_key"] != merged["__addr_key_cached"]))
	) if "Latitude_cached" in merged.columns else True

	lat_new, lon_new = [], []
	calls_made = 0
	for idx, row in merged.iterrows():
		need_call = (isinstance(to_query_mask, bool) and to_query_mask) or (not isinstance(to_query_mask, bool) and to_query_mask.loc[idx])
		if not do_api or not need_call or calls_made >= remaining:
			lat_new.append(row.get("Latitude_cached", np.nan))
			lon_new.append(row.get("Longitude_cached", np.nan))
			continue
		if (not force) and pd.notna(row.get("Latitude_cached")) and pd.notna(row.get("Longitude_cached")):
			lat_new.append(row["Latitude_cached"]); lon_new.append(row["Longitude_cached"])
			continue
		res = _google_lookup(row["__addr_key"], api_key)
		if res:
			lat_new.append(res[0]); lon_new.append(res[1])
		else:
			lat_new.append(row.get("Latitude_cached", np.nan)); lon_new.append(row.get("Longitude_cached", np.nan))
		calls_made += 1
		time.sleep(0.10)

	if calls_made:
		month_usage["count"] = month_usage.get("count", 0) + calls_made
		_save_month_usage(datetime.utcnow())

	merged["Latitude_final"] = lat_new
	merged["Longitude_final"] = lon_new
	update_rows = merged[["CCN","Latitude_final","Longitude_final"]].rename(columns={"Latitude_final":"Latitude","Longitude_final":"Longitude"})
	update_rows = update_rows.merge(sub[["CCN","Address","City","State","Zip Code","Year"]], on="CCN", how="left")
	update_rows["__has_coords"] = update_rows["Latitude"].notna() & update_rows["Longitude"].notna()
	update_rows = (update_rows
				   .sort_values(["CCN","__has_coords","Year"], ascending=[True,False,False])
				   .drop_duplicates(subset=["CCN"], keep="first")
				   .drop(columns="__has_coords"))

	# Replace cache update block with preservation logic
	# Ensure we do not overwrite existing populated coords with NaNs
	existing_cache = _load_geocode_cache()
	if not existing_cache.empty:
		existing_cache["CCN"] = existing_cache["CCN"].astype(str).str.strip()
	update_rows["CCN"] = update_rows["CCN"].astype(str).str.strip()
	if existing_cache.empty:
		cache_out = update_rows
	else:
		combined = pd.concat([existing_cache, update_rows], ignore_index=True)
		combined["__has_coords"] = combined["Latitude"].notna() & combined["Longitude"].notna()
		combined = combined.sort_values(["CCN","__has_coords"], ascending=[True, False])
		# Keep last occurrence with coords; if none have coords keep last anyway
		cache_out = (combined
					 .drop_duplicates(subset=["CCN"], keep="first")
					 .drop(columns="__has_coords"))
	try:
		cache_out.to_csv(_GOOGLE_GEOCODE_CSV, index=False)
	except Exception as e:
		logger.warning(f"Failed to persist geocode cache: {e}")
	# Build final_map with unique CCNs only
	cache_unique = cache_out.drop_duplicates(subset=["CCN"], keep="last")
	final_map = cache_unique.set_index("CCN")[["Latitude","Longitude"]].to_dict("index")
	def _apply(r):
		ccn = str(r["CCN"]).strip()
		if ccn in final_map and (pd.isna(r["Latitude"]) or pd.isna(r["Longitude"])):
			return final_map[ccn]["Latitude"], final_map[ccn]["Longitude"]
		return r["Latitude"], r["Longitude"]
	base_df["Latitude"], base_df["Longitude"] = zip(*base_df.apply(_apply, axis=1))
	return base_df

def _compute_potential_cah(df: pd.DataFrame) -> pd.DataFrame:
	required = ["CCN","Facility Type","Rural vs. Urban","Number Beds","Latitude","Longitude"]
	for c in required:
		if c not in df.columns:
			df[c] = pd.NA
	dedup = df.drop_duplicates(subset=["CCN"]).dropna(subset=["Latitude","Longitude"]).copy()
	if dedup.empty:
		df["Min_Distance_Miles"] = 0.0
		df["Potential_CAH_Google"] = 0
		return df
	lat = pd.to_numeric(dedup["Latitude"], errors="coerce")
	lon = pd.to_numeric(dedup["Longitude"], errors="coerce")
	valid = lat.notna() & lon.notna()
	dedup = dedup.loc[valid].copy()
	lat_rad = np.radians(lat[valid].to_numpy())
	lon_rad = np.radians(lon[valid].to_numpy())
	if len(dedup) == 0:
		df["Min_Distance_Miles"] = 0.0
		df["Potential_CAH_Google"] = 0
		return df
	dlat = lat_rad[:, None] - lat_rad[None, :]
	dlon = lon_rad[:, None] - lon_rad[None, :]
	a = np.sin(dlat/2)**2 + np.cos(lat_rad[:, None]) * np.cos(lat_rad[None, :]) * np.sin(dlon/2)**2
	c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
	dist = 3958.795 * c
	np.fill_diagonal(dist, 0.0)
	min_dist = []
	for i in range(dist.shape[0]):
		eligible = dist[i][dist[i] >= 1.0]
		min_dist.append(float(np.min(eligible)) if eligible.size else 0.0)
	dedup["Min_Distance_Miles"] = min_dist
	# Temporarily DISABLED beds <= 25 requirement per review request.
	# Original clause:
	#    & (pd.to_numeric(dedup["Number Beds"], errors="coerce") <= 25)
	dedup["Potential_CAH_Google"] = np.where(
		(dedup["Min_Distance_Miles"] > 35) &
		(dedup["Facility Type"] != "CAH") &
		(dedup["Rural vs. Urban"] == "Rural"),
		1, 0
	)
	dist_map = dedup.set_index("CCN")["Min_Distance_Miles"].to_dict()
	flag_map = dedup.set_index("CCN")["Potential_CAH_Google"].to_dict()
	df["Min_Distance_Miles"] = df["CCN"].map(dist_map).fillna(0.0)
	df["Potential_CAH_Google"] = df["CCN"].map(flag_map).fillna(0).astype(int)
	return df

def _resolve_duplicate_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
	for col in list(df.columns):
		if col.endswith("_x"):
			base = col[:-2]
			other = base + "_y"
			if other in df.columns:
				df[base] = df[col].combine_first(df[other])
				df = df.drop(columns=[col, other])
	return df

def _recompute_ebitdar(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Recompute EBITDAR if component columns (Net Income, Interest, Taxes, Depreciation, Rent) are all present
	and at least one has a non-null value. Keeps existing EBITDAR if components are absent or all null.
	EBITDAR = Net Income + Interest + Taxes + Depreciation + Rent
	"""
	required = ['Net Income','Interest','Taxes','Depreciation','Rent']
	if not all(c in df.columns for c in required):
		return df
	comp = df[required].apply(pd.to_numeric, errors='coerce')
	if comp.isna().all(axis=None):
		return df  # nothing to recompute
	new_ebitdar = comp.sum(axis=1, min_count=1)
	# Only replace EBITDAR where recomputation yields a non-null and existing EBITDAR is null
	if 'EBITDAR' not in df.columns:
		df['EBITDAR'] = new_ebitdar
	else:
		existing = pd.to_numeric(df['EBITDAR'], errors='coerce')
		df.loc[existing.isna() & new_ebitdar.notna(), 'EBITDAR'] = new_ebitdar
	return df

def _compute_ebitdar_components(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Recompute EBITDAR only if Interest, Taxes, Depreciation, Rent exist with data.
	If components missing, leave existing EBITDAR untouched.
	"""
	required = ["Net Income","Interest","Taxes","Depreciation","Rent"]
	if all(c in df.columns for c in required):
		# Check if any component has at least one non-null
		if any(df[c].notna().any() for c in required):
			comp = {c: pd.to_numeric(df[c], errors="coerce").fillna(0) for c in required}
			recalc = comp["Net Income"] + comp["Interest"] + comp["Taxes"] + comp["Depreciation"] + comp["Rent"]
			if "EBITDAR" not in df.columns:
				df["EBITDAR"] = recalc
			else:
				df["EBITDAR"] = np.where(df["EBITDAR"].isna(), recalc, df["EBITDAR"])
	return df

def _attach_financial_metrics(df: pd.DataFrame) -> pd.DataFrame:
	# Only try to recompute EBITDAR if components truly present
	df = _compute_ebitdar_components(df)
	for c in ['Total Current Liabilities','Total Long-Term Liabilities','Cash',
			  'Operating Expense','Other Expense','Government Reimbursements',
			  'Net Receivables','Total Current Assets','Net Income','EBITDAR']:
		if c not in df.columns:
			df[c] = np.nan
	# Total Debt
	df['Total Debt'] = pd.to_numeric(df['Total Current Liabilities'], errors='coerce').fillna(0) + \
					   pd.to_numeric(df['Total Long-Term Liabilities'], errors='coerce').fillna(0)
	# Net Income Average per CCN
	if 'CCN' in df.columns:
		ni = pd.to_numeric(df['Net Income'], errors='coerce')
		ni_avg = pd.DataFrame({'CCN': df['CCN'], 'NI': ni}).groupby('CCN')['NI'].mean()
		df['Net Income Average'] = df['CCN'].map(ni_avg)
	else:
		df['Net Income Average'] = np.nan
	# Flags
	ebitdar_num = pd.to_numeric(df['EBITDAR'], errors='coerce')
	ni_avg_num = pd.to_numeric(df['Net Income Average'], errors='coerce')
	cash = pd.to_numeric(df['Cash'], errors='coerce')
	oper = pd.to_numeric(df['Operating Expense'], errors='coerce')
	other = pd.to_numeric(df['Other Expense'], errors='coerce')
	cur_assets = pd.to_numeric(df['Total Current Assets'], errors='coerce')
	gov = pd.to_numeric(df['Government Reimbursements'], errors='coerce')
	recv = pd.to_numeric(df['Net Receivables'], errors='coerce')
	total_exp = oper + other

	df['>$400k EBITDAR'] = (ebitdar_num > 400_000).astype(int)
	df['Debt <$5M or 5x NI'] = ((df['Total Debt'] < 5_000_000) | (df['Total Debt'] < 5 * ni_avg_num)).astype(int)
	df['Positive Cash Position'] = (cash > 0).astype(int)
	df['Cash vs. 65 Days'] = (cash > (oper / 365.0 * 65)).astype(int)
	df['Income vs. 3% Total Expenses'] = (pd.to_numeric(df['Net Income'], errors='coerce') > 0.03 * total_exp).astype(int)
	df['Assets vs. 100 Days'] = (cur_assets > (oper / 365.0 * 100)).astype(int)
	df['$100k Reimbursement'] = (gov > 100_000).astype(int)
	df['Net Receivables vs. 88 Days'] = (recv > (oper / 365.0 * 88)).astype(int)
	return df

def _update_base_with_year(base_df: pd.DataFrame, new_year_df: pd.DataFrame, year: int) -> pd.DataFrame:
	base_df = base_df.copy()
	if "Year" in base_df.columns:
		base_df["Year"] = pd.to_numeric(base_df["Year"], errors="coerce").astype("Int64")
	nyd = new_year_df.copy()
	if "Year" in nyd.columns:
		nyd["Year"] = pd.to_numeric(nyd["Year"], errors="coerce").fillna(year).astype(int)
	else:
		nyd["Year"] = int(year)
	nyd = nyd[nyd["Year"] == int(year)].copy()
	if "Zip Code" in nyd.columns:
		nyd["Zip Code"] = nyd["Zip Code"].apply(_clean_zip)
	for c in ["CCN","Hospital Name"]:
		if c not in nyd.columns:
			nyd[c] = pd.NA
	nyd = nyd[~(nyd["CCN"].isna() & nyd["Hospital Name"].isna())]
	nyd = _dedupe_core(nyd)
	if "Year" in base_df.columns:
		base_df = base_df[base_df["Year"] != int(year)]
	combined = pd.concat([base_df, nyd], ignore_index=True)
	combined = _dedupe_core(combined)
	combined = _attach_financial_metrics(combined)
	combined = _resolve_duplicate_metric_columns(combined)
	combined = _strip_visuals_columns(combined)
	return combined

def _cleanup_build_odds_columns(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Collapse multiple Build_Odds* columns into one 'Build_Odds' picking:
	1. Existing 'Build_Odds' if present.
	2. Else the column with most non-null values.
	Removes all other Build_Odds* variants.
	"""
	candidates = [c for c in df.columns if c.startswith("Build_Odds")]
	if not candidates:
		return df
	if "Build_Odds" in candidates:
		main = df["Build_Odds"]
	else:
		best = max(candidates, key=lambda c: df[c].notna().sum())
		main = df[best]
	df = df.drop(columns=candidates)
	df["Build_Odds"] = main
	return df

def _get_metric_columns(df: pd.DataFrame) -> list[str]:
	"""Return list of financial / feature columns if present."""
	candidates = [
		"Report Record Number","Discharges","Cost of Charity","Cost of Uncompensated Care","Net Income",
		"Interest","Taxes","Depreciation","Rent","EBITDAR","Total Current Liabilities","Total Long-Term Liabilities",
		"Cash","Operating Expense","Other Expense","Government Reimbursements","Net Receivables","Total Current Assets",
		"Total Debt","Net Income Average",">$400k EBITDAR","Debt <$5M or 5x NI","Positive Cash Position",
		"Cash vs. 65 Days","Income vs. 3% Total Expenses","Assets vs. 100 Days","$100k Reimbursement",
		"Net Receivables vs. 88 Days"
	]
	return [c for c in candidates if c in df.columns]

def _compute_build_odds_recent(base_df: pd.DataFrame,
							   recent_df: pd.DataFrame,
							   threshold_debt: float) -> pd.DataFrame:
	"""
	Train model on historical rows excluding recent CCNs; predict Build_Odds for recent_df.
	Fallback heuristic if model or features unavailable.
	"""
	if recent_df.empty:
		return recent_df
	recent_df = _ensure_model_feature_columns(recent_df)
	base_df = _ensure_model_feature_columns(base_df)
	recent_df["Build_Odds"] = np.nan
	exclude_ccns = set(recent_df["CCN"])
	historical_train = base_df[~base_df["CCN"].isin(exclude_ccns)].copy()
	if historical_train.empty:
		logger.warning("Build odds: historical training set empty.")
		return recent_df
	feature_cols = [c for c in _get_metric_columns(historical_train)
					if pd.api.types.is_numeric_dtype(historical_train[c])]
	# Add encoded features
	for enc in ["RuralUrbanNum","Number Beds","TypeControlNum"]:
		if enc in historical_train.columns and enc not in feature_cols:
			feature_cols.append(enc)
	if not feature_cols:
		logger.warning("Build odds: no numeric feature columns found.")
		return recent_df
	X_train = historical_train[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
	X_recent = recent_df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
	try:
		if train_debt_model is None:
			raise RuntimeError("train_debt_model unavailable")
		model, meta = train_debt_model(historical_train, threshold=threshold_debt)
		logger.info(f"Build odds: trained model; features={len(feature_cols)} train_rows={len(X_train)}")
		if hasattr(model, "predict_proba"):
			probs = model.predict_proba(X_recent)
			if probs.ndim == 2 and probs.shape[1] > 1:
				recent_df["Build_Odds"] = probs[:,1]
			else:
				recent_df["Build_Odds"] = probs.flatten()
		elif hasattr(model, "predict"):
			preds = model.predict(X_recent)
			preds = pd.to_numeric(pd.Series(preds), errors='coerce')
			if preds.min() < 0 or preds.max() > 1:
				r = preds.max() - preds.min()
				recent_df["Build_Odds"] = (preds - preds.min()) / r if r else 0.0
			else:
				recent_df["Build_Odds"] = preds
		else:
			raise RuntimeError("Model lacks prediction interface")
	except Exception as e:
		logger.warning(f"Build odds model failed ({e}); using heuristic.")
		flag_cols = [c for c in recent_df.columns
					 if recent_df[c].dropna().isin([0,1]).all() and recent_df[c].nunique() <= 3]
		if flag_cols:
			tmp = recent_df[flag_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
			recent_df["Build_Odds"] = (tmp.sum(axis=1) / len(flag_cols)).clip(0,1)
		else:
			recent_df["Build_Odds"] = 0.5
	logger.info(f"Build odds: predictions generated rows={recent_df['Build_Odds'].notna().sum()}")
	return recent_df

def _ensure_model_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
	"""Ensure ML-required categorical features exist & encoded."""
	df = df.copy()
	if "Rural vs. Urban" not in df.columns: df["Rural vs. Urban"] = pd.NA
	df["RuralUrbanNum"] = df["Rural vs. Urban"].replace({"Rural":1,"Urban":0}).apply(lambda v: 1 if v==1 else (0 if v==0 else np.nan))
	if "Number Beds" not in df.columns: df["Number Beds"] = np.nan
	df["Number Beds"] = pd.to_numeric(df["Number Beds"], errors="coerce")
	if "Type of Control" not in df.columns: df["Type of Control"] = pd.NA
	ctrl_map = {'Non-Profit':1,'Private':2,'Government - Federal':3,'Government - City-County':4,'Government - County':5,'Government - State':6,'Government - City':7,'Government':8}
	df["TypeControlNum"] = df["Type of Control"].map(ctrl_map)
	return df

def _compute_potential_cah_dist(dedup: pd.DataFrame) -> pd.DataFrame:
	d = dedup.copy()
	for c in ["Latitude","Longitude","Facility Type","Rural vs. Urban","Number Beds","CCN"]:
		if c not in d.columns: d[c] = pd.NA
	d["Latitude"] = pd.to_numeric(d["Latitude"], errors="coerce")
	d["Longitude"] = pd.to_numeric(d["Longitude"], errors="coerce")
	d = d.dropna(subset=["Latitude","Longitude"])
	if d.empty:
		dedup["Potential_CAH_Google"] = np.nan
		return dedup
	lat = np.radians(d["Latitude"].to_numpy())
	lon = np.radians(d["Longitude"].to_numpy())
	dlat = lat[:,None]-lat[None,:]
	dlon = lon[:,None]-lon[None,:]
	a = np.sin(dlat/2)**2 + np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin(dlon/2)**2
	c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
	dist = 3958.795 * c
	np.fill_diagonal(dist, 0.0)
	min_dist = []
	for i in range(dist.shape[0]):
		eligible = dist[i][dist[i] >= 1.0]  # ignore same campus <1 mile
		min_dist.append(float(np.min(eligible)) if eligible.size else 0.0)
	d["Min_Distance_Miles"] = min_dist
	# Beds constraint disabled per approval
	d["Potential_CAH_Google"] = np.where(
		(d["Facility Type"] != "CAH") &
		(d["Rural vs. Urban"] == "Rural") &
		(d["Min_Distance_Miles"] > 35),
		1, 0
	)
	flag_map = d.set_index("CCN")["Potential_CAH_Google"].to_dict()
	dedup["Potential_CAH_Google"] = dedup["CCN"].map(flag_map)
	return dedup

def _compute_build_odds_recent(base_df: pd.DataFrame, recent_df: pd.DataFrame, threshold_debt: float) -> pd.DataFrame:
	r = recent_df.copy()
	r["Build_Odds"] = np.nan

	# Ensure alpha core columns first
	base_df = _ensure_alpha_core_columns(base_df)
	r = _ensure_alpha_core_columns(r, reference=base_df)

	# Existing model feature encoding
	base_df = _ensure_model_feature_columns(base_df)
	r = _ensure_model_feature_columns(r)

	# Hard guard: required alpha columns must be present in both
	alpha_missing = [c for c in ["Type of Control","Rural vs. Urban"] if c not in base_df.columns or c not in r.columns]
	if alpha_missing:
		logger.warning(f"Build odds skipped (alpha columns missing): {alpha_missing}")
		return r

	missing = [c for c in _REQUIRED_MODEL_COLS if c not in base_df.columns or c not in r.columns]
	if missing:
		logger.warning(f"Build odds skipped (missing features after enforcement): {missing}")
		return r

	exclude = set(r["CCN"])
	train = base_df[~base_df["CCN"].isin(exclude)].copy()
	if train.empty:
		logger.warning("Build odds skipped (empty training set).")
		return r
	if train_debt_model is None:
		logger.warning("Build odds skipped (train_debt_model unavailable).")
		return r

	try:
		model, _ = train_debt_model(train, threshold=threshold_debt)
		if not hasattr(model, "predict_proba"):
			logger.warning("Build odds skipped (model lacks predict_proba).")
			return r
		feature_cols = [c for c in train.columns
						if c not in {"CCN","Hospital Name","Year"} and pd.api.types.is_numeric_dtype(train[c])]
		if not feature_cols:
			logger.warning("Build odds skipped (no numeric feature columns).")
			return r
		X_recent = r[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
		r["Build_Odds"] = model.predict_proba(X_recent)[:,1]
		logger.info(f"Build odds predictions rows={r['Build_Odds'].notna().sum()} features={len(feature_cols)}")
	except Exception as e:
		logger.warning(f"Build odds skipped (model error): {e}")
	return r

def _ensure_alpha_core_columns(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
	"""
	Guarantee presence and propagation of 'Type of Control', 'Rural vs. Urban', 'Number Beds'.
	If reference provided, map by CCN; else leave existing or fill NA.
	"""
	df = df.copy()
	ref = reference if reference is not None else df
	ccn_present = "CCN" in df.columns and "CCN" in ref.columns
	for col in ["Type of Control","Rural vs. Urban","Number Beds"]:
		if col not in df.columns or df[col].isna().all():
			if col in ref.columns and ccn_present:
				map_vals = ref.dropna(subset=[col]).drop_duplicates("CCN").set_index("CCN")[col].to_dict()
				df[col] = df["CCN"].map(map_vals)
			elif col in ref.columns:
				df[col] = ref[col]
			else:
				df[col] = pd.NA
	# Normalize 'Rural vs. Urban' textual values (if numeric codes slipped in)
	if "Rural vs. Urban" not in df.columns:
		df["Rural vs. Urban"] = pd.NA
	# Normalize textual values
	if "Rural vs. Urban" in df.columns:
		df["Rural vs. Urban"] = (
			df["Rural vs. Urban"]
			.replace({1: "Urban", 2: "Rural"})
			.where(df["Rural vs. Urban"].isin(["Urban","Rural"]), df["Rural vs. Urban"])
		)
	return df

# Geocode cache & batch paths
if '_GOOGLE_GEOCODE_CSV' not in globals():
	_GOOGLE_GEOCODE_CSV = Path(__file__).parent / "google_geocoded.csv"
if '_GEOCODE_BATCH_DIR' not in globals():
	_GEOCODE_BATCH_DIR = Path(__file__).parent / "geocode_temp"

# Geocode readiness (fix missing constant reference)
if '_geocode_ready' not in globals():
	def _geocode_ready() -> bool:
		try:
			_GEOCODE_BATCH_DIR.mkdir(parents=True, exist_ok=True)
			t = _GEOCODE_BATCH_DIR / "__test.tmp"
			t.write_text("ok", encoding="utf-8")
			t.unlink(missing_ok=True)
			_GOOGLE_GEOCODE_CSV.parent.mkdir(parents=True, exist_ok=True)
			tc = _GOOGLE_GEOCODE_CSV.parent / "__cache_test.tmp"
			tc.write_text("ok", encoding="utf-8")
			tc.unlink(missing_ok=True)
			return True
		except Exception as e:
			logger.warning(f"Geocode disabled (persistence not writable): {e}")
			return False

# Ensure cache append helper exists
if '_append_geocode_cache' not in globals():
	def _append_geocode_cache(new_rows: pd.DataFrame):
		if new_rows.empty: return
		try:
			existing = pd.read_csv(_GOOGLE_GEOCODE_CSV) if _GOOGLE_GEOCODE_CSV.exists() else pd.DataFrame(columns=["CCN","Address","City","State","Zip Code","Latitude","Longitude"])
		except Exception:
			existing = pd.DataFrame(columns=["CCN","Address","City","State","Zip Code","Latitude","Longitude"])
		all_rows = pd.concat([existing, new_rows], ignore_index=True)
		all_rows["CCN"] = all_rows["CCN"].astype(str).str.strip()
		# keep last non-null coords
		all_rows = (all_rows
			.sort_values(["CCN","Latitude","Longitude"])
			.drop_duplicates(subset=["CCN"], keep="last"))
		try:
			all_rows.to_csv(_GOOGLE_GEOCODE_CSV, index=False)
			logger.info(f"Geocode cache updated total={len(all_rows)} missing={(all_rows['Latitude'].isna()|all_rows['Longitude'].isna()).sum()}")
		except Exception as e:
			logger.error(f"Failed writing geocode cache: {e}")

# Define geocode function if missing
if '_geocode_missing' not in globals():
	def _geocode_missing(df: pd.DataFrame, api_key: str, max_calls: int = 10000) -> pd.DataFrame:
		out = df.copy()
		for c in ["Address","City","State","Zip Code","CCN","Latitude","Longitude"]:
			if c not in out.columns:
				out[c] = "" if c not in ("Latitude","Longitude") else np.nan
		missing = out[out["Latitude"].isna() | out["Longitude"].isna()]
		total_missing = len(missing)
		if total_missing == 0:
			logger.info("Geocode: nothing missing.")
			return out
		if not _geocode_ready():
			logger.info("Geocode: persistence not ready, skipping.")
			return out
		actual_limit = min(max_calls, total_missing)
		logger.info(f"Geocode: attempting={actual_limit} of missing={total_missing} (env limit={max_calls}).")
		batch_rows = []
		success_total = 0
		calls = 0
		batch_index = 0
		for _, r in missing.iterrows():
			if calls >= actual_limit:
				break
			addr = f"{str(r['Address']).strip()}, {str(r['City']).strip()}, {str(r['State']).strip()}, {str(r['Zip Code']).strip()}"
			try:
				resp = requests.get("https://maps.googleapis.com/maps/api/geocode/json",
									params={"address": addr, "key": api_key},
									timeout=8)
				if resp.status_code == 200:
					j = resp.json()
					if j.get("status") == "OK" and j.get("results"):
						loc = j["results"][0]["geometry"]["location"]
						batch_rows.append({
							"CCN": _norm_ccn(r["CCN"]),
							"Address": r["Address"],
							"City": r["City"],
							"State": r["State"],
							"Zip Code": r["Zip Code"],
							"Latitude": loc["lat"],
							"Longitude": loc["lng"]
						})
						success_total += 1
			except Exception:
				pass
			calls += 1
			if calls % 250 == 0:
				batch_index += 1
				bpath = _GEOCODE_BATCH_DIR / f"geocode_batch_{batch_index}.csv"
				try:
					pd.DataFrame(batch_rows).to_csv(bpath, index=False)
					logger.info(f"Geocode batch persisted batch={batch_index} successes={len(batch_rows)}")
					batch_rows.clear()
				except Exception as e:
					logger.error(f"Batch write failed batch={batch_index} error={e}")
				logger.info(f"Geocode progress calls={calls} total_success={success_total}")
			time.sleep(0.10)
		if batch_rows:
			batch_index += 1
			bpath = _GEOCODE_BATCH_DIR / f"geocode_batch_{batch_index}.csv"
			try:
				pd.DataFrame(batch_rows).to_csv(bpath, index=False)
				logger.info(f"Geocode final batch persisted batch={batch_index} successes={len(batch_rows)}")
			except Exception as e:
				logger.error(f"Final batch write failed: {e}")
			batch_rows.clear()
		logger.info(f"Geocode complete calls={calls} successes={success_total}")
		# Consolidate
		try:
			batches = sorted(_GEOCODE_BATCH_DIR.glob("geocode_batch_*.csv"))
			if batches:
				combined = pd.concat([pd.read_csv(p) for p in batches if p.stat().st_size > 0], ignore_index=True)
				_append_geocode_cache(combined)
				if os.getenv("CMS_KEEP_GEOCODE_BATCH","0") != "1":
					for p in batches:
						p.unlink(missing_ok=True)
				logger.info(f"Consolidated {len(batches)} batch file(s) into cache.")
		except Exception as e:
			logger.error(f"Batch consolidation failed: {e}")
		return _map_cached_geocodes(out)

# ---------- Functions ----------

def _http_get(url: str, params: dict | None = None, timeout: int = 60) -> tuple[int, bytes | None]:
	"""Simple HTTP GET with timeout; return (status_code, content_bytes)."""
	try:
		r = requests.get(url, params=params, timeout=timeout)
		return r.status_code, r.content
	except Exception:
		return 0, None

def _download_hcris_zip(year: int, out_dir: Path) -> Path | None:
	"""Download and extract HCRIS ZIP file for given year."""
	out_dir.mkdir(parents=True, exist_ok=True)
	url = f"https://downloads.cms.gov/FILES/HCRIS/HOSP10FY{year}.ZIP"
	status, content = _http_get(url)
	if status != 200 or not content:
		logger.warning(f"HCRIS download failed year={year} status={status}")
		return None
	try:
		with zipfile.ZipFile(BytesIO(content)) as zf:
			zf.extractall(out_dir)
			logger.info(f"HCRIS zip extracted year={year} files={len(zf.namelist())}")
		return out_dir
	except Exception as e:
		logger.warning(f"HCRIS unzip failed year={year} error={e}")
		return None

def _map_cached_geocodes(df: pd.DataFrame) -> pd.DataFrame:
	"""Attach cached lat/long to df by CCN without overwriting existing non-null coordinates."""
	cache = _load_geocode_cache() if '_load_geocode_cache' in globals() else pd.DataFrame()
	out = df.copy()
	if "CCN" not in out.columns:
		out["CCN"] = ""
	out["CCN"] = out["CCN"].apply(_norm_ccn)
	for col in ["Latitude","Longitude"]:
		if col not in out.columns:
			out[col] = np.nan
	if cache.empty or "CCN" not in cache.columns:
		return out
	cache = cache.dropna(subset=["Latitude","Longitude"])
	if cache.empty:
		return out
	cache["CCN"] = cache["CCN"].apply(_norm_ccn)
	cache = cache.drop_duplicates(subset=["CCN"], keep="last")
	out = out.merge(cache[["CCN","Latitude","Longitude"]], on="CCN", how="left", suffixes=("","_cached"))
	if "Latitude_cached" in out.columns:
		out["Latitude"] = np.where(out["Latitude"].notna(), out["Latitude"], out["Latitude_cached"])
		out["Longitude"] = np.where(out["Longitude"].notna(), out["Longitude"], out["Longitude_cached"])
		out = out.drop(columns=["Latitude_cached","Longitude_cached"])
	return out

def build_full_refresh_local(folder: str | Path = '.', start_year: int = 2015, end_year: int | None = None) -> pd.DataFrame:
	"""Build full multi-year baseline locally (no geocode, no CAH, no build odds)."""
	base_dir = Path(folder).resolve()
	end_year = end_year or (datetime.utcnow().year - 1)
	all_years = []
	for y in range(start_year, end_year + 1):
		raw_dir = _download_hcris_zip(y, base_dir / f"hcris_raw_{y}")
		if raw_dir is None:
			continue
		try:
			ydf = assemble_year_from_files(raw_dir, y)
			if ydf is not None and not ydf.empty:
				all_years.append(ydf)
		except Exception:
			continue
	if not all_years:
		raise RuntimeError("No yearly data assembled.")
	full = pd.concat(all_years, ignore_index=True)
	full = full.drop_duplicates()
	full.to_csv(base_dir / "cms_data.csv", index=False)
	return full

def run_monthly_pipeline(year: int | None = None,
						 folder: str | Path = '.',
						 threshold_debt: float = 20_000_000,
						 prior_year_df: pd.DataFrame | None = None,
						 blob_connection_string: str | None = None,
						 container_name: str = "cmsfiles") -> dict:
	base_dir = MODULE_DIR if str(folder)=='.' else Path(folder).resolve()
	base_dir.mkdir(parents=True, exist_ok=True)
	target_year = year or (datetime.utcnow().year - 1)
	base_path = base_dir / BASE_BLOB_NAME
	if not base_path.exists():
		return {"error": f"Baseline {BASE_BLOB_NAME} not found. Run full refresh first."}

	try:
		baseline = pd.read_csv(base_path, low_memory=False)
	except Exception as e:
		return {"error": f"Failed to read baseline: {e}"}

	raw_dir = _download_hcris_zip(target_year, base_dir / f"hcris_raw_{target_year}")
	if raw_dir is None:
		return {"error": f"Failed to download HCRIS for {target_year}"}

	try:
		year_df = assemble_year_from_files(raw_dir, target_year)
		master_year = assemble_master_incremental(baseline, year_df, target_year)
	except Exception as e:
		return {"error": f"Year assembly failed: {e}"}
	if master_year is None or master_year.empty:
		return {"error": f"No data parsed for {target_year}"}

	# Ensure Year column is correct
	if "Year" in master_year.columns:
		master_year["Year"] = pd.to_numeric(master_year["Year"], errors="coerce").fillna(target_year).astype(int)
		master_year = master_year[master_year["Year"] == target_year]

	# Update baseline with new year rows
	updated = _update_base_with_year(baseline, master_year, target_year)
	updated = _safe_drop_columns(updated, ["Latitude","Longitude","Potential_CAH_Google","Build_Odds","Min_Distance_Miles"])
	updated = _ensure_alpha_baseline(updated, master_year)

	# Snapshot
	run_dt = datetime.utcnow()
	snapshot_name = _snapshot_name(BASE_BLOB_NAME, run_dt)
	snapshot_path = base_dir / snapshot_name
	updated.to_csv(base_path, index=False)
	updated.to_csv(snapshot_path, index=False)

	# Dedup block
	dedup_path = None
	enriched_dedup = pd.DataFrame()
	recent_years = []
	try:
		years_present = pd.to_numeric(updated["Year"], errors="coerce").dropna().astype(int).unique()
		recent_years = [y for y in [target_year, target_year - 1] if y in years_present]
	except Exception:
		recent_years = []

	if recent_years:
		recent_subset = updated[updated["Year"].isin(recent_years)].copy()
		dedup = (recent_subset
				 .sort_values(["CCN","Year"], ascending=[True, False])
				 .drop_duplicates(subset=["CCN"], keep="first"))
		dedup["CCN"] = dedup["CCN"].apply(_norm_ccn)

		# Propagate alpha columns
		dedup = _propagate_alpha_columns(dedup, updated, ["Rural vs. Urban","Type of Control","Number Beds"])
		# Ensure alpha columns explicitly present
		for col in ["Rural vs. Urban","Type of Control","Number Beds"]:
			if col not in dedup.columns:
				dedup[col] = pd.NA
		# Map existing geocodes
		dedup = _map_cached_geocodes(dedup)
		# Geocode call guarded
		api_key = os.environ.get("GOOGLE_API_KEY")
		geocode_limit = int(os.getenv("CMS_GEOCODE_MAX_CALLS", "10000"))
		if api_key and _geocode_ready() and geocode_limit > 0:
			logger.info(f"Geocode max_calls={geocode_limit}")
			dedup = _geocode_missing(dedup, api_key=api_key, max_calls=geocode_limit)
		else:
			logger.info("Geocode skipped (missing key, persistence not ready, or limit <= 0).")
		# CAH
		dedup = _compute_potential_cah_dist(dedup)
		# Build odds
		dedup = _compute_build_odds_recent(updated, dedup, threshold_debt)

		dedup_path = base_dir / "recent_hospitals_dedup.csv"
		dedup.to_csv(dedup_path, index=False)
		enriched_dedup = dedup

	# Merge visuals
	try:
		visuals_df = _merge_visuals(updated, enriched_dedup)
	except Exception as e:
		logger.warning(f"Visuals merge failed: {e}")
		visuals_df = updated.copy()
		for c in ["Latitude","Longitude","Potential_CAH_Google","Build_Odds"]:
			if c not in visuals_df.columns:
				visuals_df[c] = np.nan

	visuals_path = base_dir / VISUALS_BLOB_NAME
	visuals_df.to_csv(visuals_path, index=False)

	return {
		"year": target_year,
		"rows_base": len(updated),
		"rows_visuals": len(visuals_df),
		"snapshot": snapshot_name,
		"visuals_path": str(visuals_path),
		"dedup_path": str(dedup_path) if dedup_path else None,
		"recent_years_visuals": recent_years,
		"dedup_missing_geocode": int(enriched_dedup["Latitude"].isna().sum()) if not enriched_dedup.empty and "Latitude" in enriched_dedup.columns else None,
		"dedup_cah_positive": int((enriched_dedup.get("Potential_CAH_Google", pd.Series()) == 1).sum()) if not enriched_dedup.empty else None
	}

def _snapshot_name(base_name: str, dt: datetime) -> str:
	"""Return dated snapshot file name: cms_data_MM_YY.csv."""
	stamp = dt.strftime("%m_%y")
	root = base_name.rsplit(".csv", 1)[0]
	return f"{root}_{stamp}.csv"

def _safe_drop_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
	"""Drop columns if present without raising errors."""
	for c in cols:
		if c in df.columns:
			df = df.drop(columns=c)
	return df

# Ensure alpha propagation helper exists before usage
if '_propagate_alpha_columns' not in globals():
	def _propagate_alpha_columns(df: pd.DataFrame,
								 reference: pd.DataFrame,
								 cols: list[str]) -> pd.DataFrame:
		"""
		Map descriptor columns (e.g. Rural vs. Urban, Type of Control, Number Beds) from reference by CCN.
		If missing in reference or CCN absent, fill with NA.
		"""
		out = df.copy()
		if "CCN" not in out.columns or "CCN" not in reference.columns:
			for c in cols:
				if c not in out.columns:
					out[c] = pd.NA
			return out
		out["CCN"] = out["CCN"].astype(str).str.strip()
		ref = reference.copy()
		ref["CCN"] = ref["CCN"].astype(str).str.strip()
		ref_latest = (ref
			.dropna(subset=["CCN"])
			.drop_duplicates(subset=["CCN"], keep="last"))
		for c in cols:
			if c not in out.columns or out[c].isna().all():
				if c in ref_latest.columns:
					m = ref_latest.set_index("CCN")[c].to_dict()
					out[c] = out["CCN"].map(m)
			if c not in out.columns:
				out[c] = pd.NA
		return out

# Ensure geocode cache file exists early
if not _GOOGLE_GEOCODE_CSV.exists():
	_GOOGLE_GEOCODE_CSV.write_text("CCN,Address,City,State,Zip Code,Latitude,Longitude\n", encoding="utf-8")

# Add merge visuals helper if missing
if '_merge_visuals' not in globals():
	def _merge_visuals(base_df: pd.DataFrame, enriched: pd.DataFrame) -> pd.DataFrame:
		out = base_df.copy()
		for col in ["Latitude","Longitude","Potential_CAH_Google","Build_Odds"]:
			if col not in out.columns:
				out[col] = np.nan
		if enriched.empty:
			return out
		required = ["CCN","Year","Latitude","Longitude","Potential_CAH_Google","Build_Odds"]
		for c in required:
			if c not in enriched.columns:
				enriched[c] = np.nan
		enriched["CCN"] = enriched["CCN"].astype(str).str.strip()
		out["CCN"] = out["CCN"].astype(str).str.strip()
		merged = out.merge(enriched[required], on=["CCN","Year"], how="left", suffixes=("","_new"))
		for c in ["Latitude","Longitude","Potential_CAH_Google","Build_Odds"]:
			merged[c] = np.where(merged[f"{c}_new"].notna(), merged[f"{c}_new"], merged[c])
			merged = merged.drop(columns=f"{c}_new")
		recent_years = set(enriched["Year"].unique())
		merged.loc[~merged["Year"].isin(recent_years), ["Latitude","Longitude","Potential_CAH_Google","Build_Odds"]] = np.nan
		return merged

# Strengthen alpha propagation (inject blank columns if absent)
def _ensure_alpha_baseline(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	for c in ["Rural vs. Urban","Type of Control","Number Beds"]:
		if c not in df.columns:
			df[c] = pd.NA
	df = _propagate_alpha_columns(df, ref, ["Rural vs. Urban","Type of Control","Number Beds"])
	return df

def run_monthly_pipeline(year: int | None = None,
						 folder: str | Path = '.',
						 threshold_debt: float = 20_000_000,
						 prior_year_df: pd.DataFrame | None = None,
						 blob_connection_string: str | None = None,
						 container_name: str = "cmsfiles") -> dict:
	base_dir = MODULE_DIR if str(folder)=='.' else Path(folder).resolve()
	base_dir.mkdir(parents=True, exist_ok=True)
	target_year = year or (datetime.utcnow().year - 1)
	base_path = base_dir / BASE_BLOB_NAME
	if not base_path.exists():
		return {"error": f"Baseline {BASE_BLOB_NAME} not found. Run full refresh first."}

	try:
		baseline = pd.read_csv(base_path, low_memory=False)
	except Exception as e:
		return {"error": f"Failed to read baseline: {e}"}

	raw_dir = _download_hcris_zip(target_year, base_dir / f"hcris_raw_{target_year}")
	if raw_dir is None:
		return {"error": f"Failed to download HCRIS for {target_year}"}

	try:
		year_df = assemble_year_from_files(raw_dir, target_year)
		master_year = assemble_master_incremental(baseline, year_df, target_year)
	except Exception as e:
		return {"error": f"Year assembly failed: {e}"}
	if master_year is None or master_year.empty:
		return {"error": f"No data parsed for {target_year}"}

	# Ensure Year column is correct
	if "Year" in master_year.columns:
		master_year["Year"] = pd.to_numeric(master_year["Year"], errors="coerce").fillna(target_year).astype(int)
		master_year = master_year[master_year["Year"] == target_year]

	# Update baseline with new year rows
	updated = _update_base_with_year(baseline, master_year, target_year)
	updated = _safe_drop_columns(updated, ["Latitude","Longitude","Potential_CAH_Google","Build_Odds","Min_Distance_Miles"])
	updated = _ensure_alpha_baseline(updated, master_year)

	# Snapshot
	run_dt = datetime.utcnow()
	snapshot_name = _snapshot_name(BASE_BLOB_NAME, run_dt)
	snapshot_path = base_dir / snapshot_name
	updated.to_csv(base_path, index=False)
	updated.to_csv(snapshot_path, index=False)

	# Dedup block
	dedup_path = None
	enriched_dedup = pd.DataFrame()
	recent_years = []
	try:
		years_present = pd.to_numeric(updated["Year"], errors="coerce").dropna().astype(int).unique()
		recent_years = [y for y in [target_year, target_year - 1] if y in years_present]
	except Exception:
		recent_years = []

	if recent_years:
		recent_subset = updated[updated["Year"].isin(recent_years)].copy()
		dedup = (recent_subset
				 .sort_values(["CCN","Year"], ascending=[True, False])
				 .drop_duplicates(subset=["CCN"], keep="first"))
		dedup["CCN"] = dedup["CCN"].apply(_norm_ccn)

		# Propagate alpha columns
		dedup = _propagate_alpha_columns(dedup, updated, ["Rural vs. Urban","Type of Control","Number Beds"])
		# Ensure alpha columns explicitly present
		for col in ["Rural vs. Urban","Type of Control","Number Beds"]:
			if col not in dedup.columns:
				dedup[col] = pd.NA
		# Map existing geocodes
		dedup = _map_cached_geocodes(dedup)
		# Geocode call guarded
		api_key = os.environ.get("GOOGLE_API_KEY")
		geocode_limit = int(os.getenv("CMS_GEOCODE_MAX_CALLS", "10000"))
		if api_key and _geocode_ready() and geocode_limit > 0:
			logger.info(f"Geocode max_calls={geocode_limit}")
			dedup = _geocode_missing(dedup, api_key=api_key, max_calls=geocode_limit)
		else:
			logger.info("Geocode skipped (missing key, persistence not ready, or limit <= 0).")
		# CAH
		dedup = _compute_potential_cah_dist(dedup)
		# Build odds
		dedup = _compute_build_odds_recent(updated, dedup, threshold_debt)

		dedup_path = base_dir / "recent_hospitals_dedup.csv"
		dedup.to_csv(dedup_path, index=False)
		enriched_dedup = dedup

	# Merge visuals
	try:
		visuals_df = _merge_visuals(updated, enriched_dedup)
	except Exception as e:
		logger.warning(f"Visuals merge failed: {e}")
		visuals_df = updated.copy()
		for c in ["Latitude","Longitude","Potential_CAH_Google","Build_Odds"]:
			if c not in visuals_df.columns:
				visuals_df[c] = np.nan

	visuals_path = base_dir / VISUALS_BLOB_NAME
	visuals_df.to_csv(visuals_path, index=False)

	return {
		"year": target_year,
		"rows_base": len(updated),
		"rows_visuals": len(visuals_df),
		"snapshot": snapshot_name,
		"visuals_path": str(visuals_path),
		"dedup_path": str(dedup_path) if dedup_path else None,
		"recent_years_visuals": recent_years,
		"dedup_missing_geocode": int(enriched_dedup["Latitude"].isna().sum()) if not enriched_dedup.empty and "Latitude" in enriched_dedup.columns else None,
		"dedup_cah_positive": int((enriched_dedup.get("Potential_CAH_Google", pd.Series()) == 1).sum()) if not enriched_dedup.empty else None
	}
