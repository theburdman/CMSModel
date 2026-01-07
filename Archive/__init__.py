import logging
import sys
import os
import json

def _load_local_settings_and_env():
	"""Load local.settings.json then .env from package dir or project root (override only if not set)."""
	paths_json = [
		os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local.settings.json'),
		os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'local.settings.json'),
	]
	for p in paths_json:
		if os.path.isfile(p):
			try:
				with open(p, 'r', encoding='utf-8') as fh:
					cfg = json.load(fh)
				val = (cfg.get('Values') or {}).get('GOOGLE_API_KEY')
				if val and not os.environ.get('GOOGLE_API_KEY'):
					os.environ['GOOGLE_API_KEY'] = val.strip()
			except Exception:
				pass
			break
	paths_env = [
		os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'),
		os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'),
	]
	for p in paths_env:
		if os.path.isfile(p) and not os.environ.get('GOOGLE_API_KEY'):
			try:
				with open(p, 'r', encoding='utf-8') as fh:
					for line in fh:
						line=line.strip()
						if not line or line.startswith('#') or '=' not in line: continue
						k,v=line.split('=',1)
						if k.strip()=='GOOGLE_API_KEY' and v.strip() and not os.environ.get('GOOGLE_API_KEY'):
							os.environ['GOOGLE_API_KEY']=v.strip().strip('"').strip("'")
							break
			except Exception:
				pass
			break

_load_local_settings_and_env()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

logger = logging.getLogger(__name__)
def _log_api_key_status():
	logger.info(f"GOOGLE_API_KEY present={bool(GOOGLE_API_KEY)}")
_log_api_key_status()

__all__ = [
	"run_monthly_pipeline",
	"build_full_refresh_local",
	"compute_cah_candidates",
	"train_debt_model",
	"attach_build_odds",
	"assemble_year_from_files",
	"assemble_master_incremental",
	"GOOGLE_API_KEY",
]

def run_monthly_pipeline_entry(**kwargs):
	return run_monthly_pipeline(**kwargs)

def compute_cah_on_visuals_df(master_df, current_year=None):
	return compute_cah_candidates(master_df, current_year=current_year)

def retrain_and_attach_build_odds(master_df, threshold=20_000_000):
	updated, metrics = attach_build_odds(master_df, threshold=threshold)
	return updated, metrics

def _resolve_output_path(filename: str) -> str:
	"""
	Decide where to write output:
	1. If env CMS_OUTPUT_DIR set: use it.
	2. If CWD has 'CMSTimerIncremental' directory: place file inside it.
	3. Else write into CWD.
	"""
	root_env = os.getenv("CMS_OUTPUT_DIR")
	if root_env:
		os.makedirs(root_env, exist_ok=True)
		return os.path.abspath(os.path.join(root_env, filename))
	cwd = os.getcwd()
	if os.path.isdir(os.path.join(cwd, "CMSTimerIncremental")):
		return os.path.abspath(os.path.join(cwd, "CMSTimerIncremental", filename))
	return os.path.abspath(os.path.join(cwd, filename))

if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
	if '--run-monthly' in sys.argv:
		year = None
		for a in sys.argv:
			if a.startswith('--year='):
				try:
					year = int(a.split('=',1)[1])
				except Exception:
					pass
		info = run_monthly_pipeline(year=year)
		print(info)
	elif '--full-refresh-local' in sys.argv:
		df = build_full_refresh_local()
		out_path = _resolve_output_path('cms_data.csv')
		print(f"[DEBUG] cwd={os.getcwd()}")
		print(f"[DEBUG] module file={__file__}")
		print(f"[DEBUG] writing cms_data.csv to: {out_path}")
		df.to_csv(out_path, index=False)
		print(f"Full rows={len(df)} written to {out_path}")
		print(f"Exists: {os.path.exists(out_path)}")
	else:
		print("Usage: python -m CMSTimerIncremental --full-refresh-local | --run-monthly [--year=YYYY]")
