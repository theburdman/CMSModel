from __future__ import annotations
from pathlib import Path
import pandas as pd

try:
	from .metrics import extract_alpha_info, extract_numeric_metrics, assemble_cms_df
except ImportError:
	from metrics import extract_alpha_info, extract_numeric_metrics, assemble_cms_df

RAW_SCHEMA = ['Report Record Number','Worksheet Code','Line Number','Column Number','Value']

def _find_year_file(folder: Path, year: int, kind: str) -> Path:
	for p in folder.iterdir():
		if p.is_file() and p.name.lower() == f"hosp10_{year}_{kind}.csv":
			return p
	raise FileNotFoundError(f"hosp10_{year}_{kind}.csv")

def _load(folder: Path, year: int, kind: str) -> pd.DataFrame:
	path = _find_year_file(folder, year, kind)
	df = pd.read_csv(path, header=None, names=RAW_SCHEMA)
	df['Form ID'] = df['Worksheet Code'] + ',' + df['Line Number'].astype(str) + ',' + df['Column Number'].astype(str)
	df['Year'] = year
	return df

def load_alpha_year(folder: Path, year: int) -> pd.DataFrame:
	return _load(folder, year, 'alpha')

def load_nmrc_year(folder: Path, year: int) -> pd.DataFrame:
	return _load(folder, year, 'nmrc')

def assemble_year_from_files(folder: str | Path, year: int) -> pd.DataFrame:
	folder = Path(folder)
	alpha = load_alpha_year(folder, year)
	nmrc = load_nmrc_year(folder, year)
	return assemble_cms_df(extract_alpha_info(alpha), extract_numeric_metrics(nmrc))
