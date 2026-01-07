import logging
import argparse
import sys
import os

if __package__ in (None, ''):
	sys.path.append(os.path.dirname(os.path.abspath(__file__)))
	try:
		from pipeline import run_monthly_pipeline, build_full_refresh_local  # type: ignore
	except ImportError:
		raise
else:
	from .pipeline import run_monthly_pipeline, build_full_refresh_local

def _resolve_output_path(filename: str) -> str:
	root_env = os.getenv("CMS_OUTPUT_DIR")
	if root_env:
		os.makedirs(root_env, exist_ok=True)
		return os.path.abspath(os.path.join(root_env, filename))
	cwd = os.getcwd()
	if os.path.isdir(os.path.join(cwd, "CMSTimerIncremental")):
		return os.path.abspath(os.path.join(cwd, "CMSTimerIncremental", filename))
	return os.path.abspath(os.path.join(cwd, filename))

def _parse_year(arg_list):
	for a in arg_list:
		if a.startswith('--year='):
			try:
				return int(a.split('=',1)[1])
			except Exception:
				return None
	return None

def main():
	logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
	args = sys.argv[1:]
	if '--full-refresh-local' in args:
		df = build_full_refresh_local()
		out_path = _resolve_output_path('cms_data.csv')
		print(f"[DEBUG] cwd={os.getcwd()}")
		print(f"[DEBUG] writing cms_data.csv to: {out_path}")
		df.to_csv(out_path, index=False)
		print(f"Full rows={len(df)} written to {out_path}")
		print(f"Exists: {os.path.exists(out_path)}")
		return
	if '--run-monthly' in args:
		year = _parse_year(args)
		info = run_monthly_pipeline(year=year)
		print(info)
		return
	print("Usage: python -m CMSTimerIncremental --full-refresh-local | --run-monthly [--year=YYYY]")

if __name__ == '__main__':
	main()
