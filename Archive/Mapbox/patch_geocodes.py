import argparse, shutil
from pathlib import Path
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent.parent
MAPBOX_DIR = Path(__file__).parent

DEFAULT_INPUT_BASE = MODULE_DIR / "cms_data.csv"
DEFAULT_INPUT_VISUALS = MODULE_DIR / "cms_data_visuals.csv"
DEFAULT_GEOCODED = MAPBOX_DIR / "mapbox_geocoded.csv"

OUT_BASE = MODULE_DIR / "cms_data_geocoded_manual.csv"
OUT_VISUALS = MODULE_DIR / "cms_data_visuals_geocoded_manual.csv"

def _norm_ccn(val):
	if val is None:
		return ""
	if isinstance(val, float) and val.is_integer():
		return str(int(val))
	return str(val).strip()

def _validate_coords(df):
	if "Latitude" not in df.columns: df["Latitude"] = pd.NA
	if "Longitude" not in df.columns: df["Longitude"] = pd.NA
	df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
	df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
	# Basic sanity filters (USA approx bounding box)
	lat_ok = df["Latitude"].between(18, 72)
	lon_ok = df["Longitude"].between(-172, -50)
	df.loc[~lat_ok, "Latitude"] = pd.NA
	df.loc[~lon_ok, "Longitude"] = pd.NA
	return df

def apply_manual_geocodes(geocode_csv: Path,
						  base_csv: Path,
						  visuals_csv: Path | None = None,
						  force: bool = False) -> dict:
	if not geocode_csv.exists():
		raise FileNotFoundError(f"Geocode file not found: {geocode_csv}")
	if not base_csv.exists():
		raise FileNotFoundError(f"Base file not found: {base_csv}")

	geo = pd.read_csv(geocode_csv)
	geo["CCN"] = geo["CCN"].apply(_norm_ccn)
	geo = _validate_coords(geo)
	geo = geo.dropna(subset=["Latitude","Longitude"]).drop_duplicates(subset=["CCN"])

	base = pd.read_csv(base_csv, low_memory=False)
	base["CCN"] = base["CCN"].apply(_norm_ccn)

	# Backup originals
	shutil.copy2(base_csv, base_csv.with_suffix(".csv.bak"))
	if visuals_csv and visuals_csv.exists():
		shutil.copy2(visuals_csv, visuals_csv.with_suffix(".csv.bak"))

	def _merge_coords(target: pd.DataFrame) -> pd.DataFrame:
		target["CCN"] = target["CCN"].apply(_norm_ccn)
		if "Latitude" not in target.columns: target["Latitude"] = pd.NA
		if "Longitude" not in target.columns: target["Longitude"] = pd.NA
		m = target.merge(geo[["CCN","Latitude","Longitude"]], on="CCN", how="left", suffixes=("","_geo"))
		if force:
			# Overwrite always where geo has data
			mask = m["Latitude_geo"].notna() & m["Longitude_geo"].notna()
		else:
			# Only fill if original missing
			mask = (m["Latitude"].isna() | m["Longitude"].isna()) & m["Latitude_geo"].notna() & m["Longitude_geo"].notna()
		m.loc[mask, "Latitude"] = m.loc[mask, "Latitude_geo"]
		m.loc[mask, "Longitude"] = m.loc[mask, "Longitude_geo"]
		m = m.drop(columns=["Latitude_geo","Longitude_geo"])
		return m

	base_out = _merge_coords(base)
	base_out.to_csv(OUT_BASE, index=False)

	visuals_out_written = False
	if visuals_csv and visuals_csv.exists():
		visuals = pd.read_csv(visuals_csv, low_memory=False)
		visuals_out = _merge_coords(visuals)
		visuals_out.to_csv(OUT_VISUALS, index=False)
		visuals_out_written = True

	return {
		"geocode_rows_used": len(geo),
		"base_rows": len(base_out),
		"visuals_rows": len(visuals_out) if visuals_out_written else 0,
		"base_output": str(OUT_BASE),
		"visuals_output": str(OUT_VISUALS) if visuals_out_written else None,
		"force_mode": force
	}

def parse_args():
	p = argparse.ArgumentParser(description="Apply manually edited geocodes to CMS datasets.")
	p.add_argument("--geocodes", default=str(DEFAULT_GEOCODED), help="Edited geocode CSV (Mapbox/).")
	p.add_argument("--base", default=str(DEFAULT_INPUT_BASE), help="cms_data.csv path.")
	p.add_argument("--visuals", default=str(DEFAULT_INPUT_VISUALS), help="cms_data_visuals.csv path (optional).")
	p.add_argument("--force", action="store_true", help="Overwrite existing lat/lon even if present.")
	return p.parse_args()

def main():
	a = parse_args()
	visuals_path = Path(a.visuals)
	if not visuals_path.exists():
		visuals_path = None
	info = apply_manual_geocodes(Path(a.geocodes), Path(a.base), visuals_path, force=a.force)
	print(info)

if __name__ == "__main__":
	main()
