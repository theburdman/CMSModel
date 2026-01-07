import pandas as pd
import numpy as np
from pathlib import Path

def _norm_ccn(s):
	if s is None:
		return ""
	s = str(s).strip()
	# Strip decimals like '123456.0'
	if s.endswith('.0'):
		s = s[:-2]
	return s

def main():
	visuals_path = Path('cms_data_visuals.csv')
	geocode_path = Path('google_geocoded.csv')

	if not visuals_path.exists():
		raise FileNotFoundError("cms_data_visuals.csv not found (run monthly pipeline first).")

	visuals = pd.read_csv(visuals_path)
	visuals['Year'] = pd.to_numeric(visuals.get('Year'), errors='coerce')
	if 'CCN' in visuals.columns:
		visuals['CCN'] = visuals['CCN'].apply(_norm_ccn)

	# Determine recent subset (current + prior year if present)
	years = visuals['Year'].dropna().astype(int).unique()
	recent_years = sorted(years)[-2:] if len(years) >= 2 else sorted(years)
	recent_df = visuals[visuals['Year'].isin(recent_years)].copy()
	recent_df = recent_df.sort_values(['CCN','Year'], ascending=[True, False]).drop_duplicates('CCN')

	# Attach coordinates if missing
	if ('Latitude' not in recent_df.columns or 'Longitude' not in recent_df.columns or
		recent_df['Latitude'].isna().all() or recent_df['Longitude'].isna().all()):

		if not geocode_path.exists():
			print("google_geocoded.csv not found; cannot attach coordinates.")
			print(f"Recent rows={len(recent_df)} CCNs without coords cannot compute distance matrix.")
			return

		geo = pd.read_csv(geocode_path)
		if 'CCN' in geo.columns:
			geo['CCN'] = geo['CCN'].apply(_norm_ccn)
			geo = geo[['CCN','Latitude','Longitude']]
			recent_df = recent_df.merge(geo, on='CCN', how='left')
		else:
			print("google_geocoded.csv lacks CCN column; cannot join.")
			return

	# Coerce numeric coordinates
	if 'Latitude' not in recent_df.columns or 'Longitude' not in recent_df.columns:
		print("Latitude/Longitude columns still missing after merge.")
		return

	recent_df['Latitude'] = pd.to_numeric(recent_df['Latitude'], errors='coerce')
	recent_df['Longitude'] = pd.to_numeric(recent_df['Longitude'], errors='coerce')
	with_coords = recent_df.dropna(subset=['Latitude','Longitude'])

	print(f"Recent years considered: {recent_years}")
	print(f"Dedup hospitals (recent): {len(recent_df)}")
	print(f"With coordinates: {len(with_coords)}")

	if len(with_coords) < 2:
		print("Need at least 2 hospitals with coordinates to compute distances; aborting.")
		return

	lat_rad = np.radians(with_coords['Latitude'].to_numpy())
	lon_rad = np.radians(with_coords['Longitude'].to_numpy())
	dlat = lat_rad[:, None] - lat_rad[None, :]
	dlon = lon_rad[:, None] - lon_rad[None, :]
	a = np.sin(dlat/2)**2 + np.cos(lat_rad[:, None]) * np.cos(lat_rad[None, :]) * np.sin(dlon/2)**2
	c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
	dist_miles = 3958.795 * c

	names = with_coords.get('Hospital Name', with_coords['CCN']).fillna(with_coords['CCN']).to_list()
	wide = pd.DataFrame(dist_miles, index=names, columns=names)

	long_form = wide.stack().reset_index()
	long_form.columns = ['Hospital_1','Hospital_2','Distance_Miles']
	long_form = long_form[long_form['Hospital_1'] != long_form['Hospital_2']]

	out = Path('distance_matrix_recent.csv')
	long_form.to_csv(out, index=False)
	print(f"Wrote {len(long_form)} pairwise rows to {out}")

if __name__ == '__main__':
	main()
