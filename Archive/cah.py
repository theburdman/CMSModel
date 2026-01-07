from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances

EARTH_RADIUS_MI = 3958.7613

def _latest_per_ccn(df: pd.DataFrame) -> pd.DataFrame:
	if 'CCN' not in df.columns or 'Year' not in df.columns:
		return df
	d = df.copy()
	d['Year'] = pd.to_numeric(d['Year'], errors='coerce')
	d = d.sort_values(['CCN','Year'], ascending=[True, False])
	return d.drop_duplicates('CCN')

def compute_cah_candidates(master_df: pd.DataFrame, current_year: int | None = None,
                           distance_threshold: float = 35.0, beds_limit: int = 25,
                           min_exclude: float = 1.0) -> pd.DataFrame:
	df = master_df.copy()
	# Ensure latitude/longitude columns exist
	if 'Latitude' not in df.columns:
		df['Latitude'] = np.nan
	if 'Longitude' not in df.columns:
		df['Longitude'] = np.nan
	if 'Year' not in df.columns:
		return df
	df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
	if current_year is None:
		current_year = int(df['Year'].max())
	scope = df[df['Year'].isin({current_year, current_year - 1})].copy()
	latest = _latest_per_ccn(scope)

	# Select rows with numeric coordinates
	lat_numeric = pd.to_numeric(latest['Latitude'], errors='coerce')
	lon_numeric = pd.to_numeric(latest['Longitude'], errors='coerce')
	valid_idx = latest.index[lat_numeric.notna() & lon_numeric.notna()]

	# Initialize output column
	if 'Nearest_Distance_Miles' not in df.columns:
		df['Nearest_Distance_Miles'] = np.nan

	if len(valid_idx) >= 2:
		coords = np.radians(np.column_stack([lat_numeric.loc[valid_idx].values,
		                                     lon_numeric.loc[valid_idx].values]))
		mat = haversine_distances(coords) * EARTH_RADIUS_MI
		np.fill_diagonal(mat, np.inf)
		mat[mat < min_exclude] = np.inf
		nearest = mat.min(axis=1)
		nearest = np.where(np.isinf(nearest), np.nan, nearest)
		df.loc[valid_idx, 'Nearest_Distance_Miles'] = nearest
	else:
		# No distances computable; leave NaN
		pass

	rural = df.get('Rural vs. Urban','').astype(str).str.lower().eq('rural')
	beds = pd.to_numeric(df.get('Number Beds'), errors='coerce')
	is_cah = df.get('Facility Type','').astype(str).str.upper() == 'CAH'
	df['Potential_CAH'] = np.where(
		rural &
		(beds <= beds_limit) &
		(pd.to_numeric(df['Nearest_Distance_Miles'], errors='coerce') >= distance_threshold) &
		(~is_cah),
		1, 0
	)
	df.loc[df['Nearest_Distance_Miles'].isna(), 'Potential_CAH'] = np.nan
	return df
