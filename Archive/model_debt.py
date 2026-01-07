from __future__ import annotations
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score

FEATURE_CATS_BASE = ['Facility Type','Type of Control','Rural vs. Urban','Network']
FEATURE_ZERO_BASE = ['Other Expense','Government Reimbursements']
FEATURE_MED_BASE = ['Number Beds','Discharges','Cost of Charity','Cost of Uncompensated Care','Net Income','EBITDAR','Total Current Liabilities','Cash','Operating Expense','Net Receivables','Total Current Assets']

def _forward_label(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
	d = df.copy()
	d['Year'] = pd.to_numeric(d['Year'], errors='coerce')
	d['Total Long-Term Liabilities'] = pd.to_numeric(d['Total Long-Term Liabilities'], errors='coerce')
	d = d.sort_values(['CCN','Year'])
	d['LTL_next'] = d.groupby('CCN')['Total Long-Term Liabilities'].shift(-1)
	d = d[~d['LTL_next'].isna()]
	d['DebtIncreaseNextYear'] = ((d['LTL_next'] - d['Total Long-Term Liabilities']) >= threshold).astype(int)
	return d

def _prep(cats, zero_cols, med_cols):
	try:
		ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
	except TypeError:
		ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
	return ColumnTransformer([
		('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', ohe)]), cats),
		('zero', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value=0.0, keep_empty_features=True))]), zero_cols),
		('med', Pipeline([('imp', SimpleImputer(strategy='median'))]), med_cols)
	])

def _normalize_features(df: pd.DataFrame, cats, zero_cols, med_cols) -> pd.DataFrame:
	d = df.replace({pd.NA: np.nan})
	for col in zero_cols + med_cols:
		if col in d.columns:
			d[col] = pd.to_numeric(d[col], errors='coerce')
	for col in cats:
		if col in d.columns:
			d[col] = d[col].astype(str).replace({'nan': np.nan, 'None': np.nan})
	return d

def _filter_features(df: pd.DataFrame):
	# Remove numeric columns that are all missing
	med_cols = [c for c in FEATURE_MED_BASE if c in df.columns]
	zero_cols = [c for c in FEATURE_ZERO_BASE if c in df.columns]
	cat_cols = [c for c in FEATURE_CATS_BASE if c in df.columns]

	empty_med = [c for c in med_cols if df[c].dropna().empty]
	med_cols = [c for c in med_cols if c not in empty_med]
	zero_var_med = [c for c in med_cols if df[c].nunique(dropna=True) <= 1]
	med_cols = [c for c in med_cols if c not in zero_var_med]

	# Zero-variance categorical
	zero_var_cat = [c for c in cat_cols if df[c].dropna().nunique() <= 1]
	cat_cols = [c for c in cat_cols if c not in zero_var_cat]

	return cat_cols, zero_cols, med_cols, {
		'dropped_empty_med': empty_med,
		'dropped_zero_var_med': zero_var_med,
		'dropped_zero_var_cat': zero_var_cat
	}

def train_debt_model(df: pd.DataFrame, threshold: float = 20_000_000, random_state: int = 42, fallback_threshold: float = 10_000_000):
	req = set(['CCN','Year','Total Long-Term Liabilities']) | set(FEATURE_CATS_BASE) | set(FEATURE_ZERO_BASE) | set(FEATURE_MED_BASE)
	missing = [c for c in req if c not in df.columns]
	if missing:
		return None, {'reason': 'missing_columns', 'missing': missing}

	def _attempt(thr):
		labeled = _forward_label(df, thr)
		if labeled.empty:
			return None
		cats, zero_cols, med_cols, drop_info = _filter_features(labeled)
		labeled = _normalize_features(labeled, cats, zero_cols, med_cols)
		X = labeled[cats + zero_cols + med_cols]
		y = labeled['DebtIncreaseNextYear'].values
		return (thr, cats, zero_cols, med_cols, drop_info, labeled, X, y)

	first = _attempt(threshold)
	# Use y array (index 7) for positive label check
	if (first is None) or (first[7].sum() == 0):
		second = _attempt(fallback_threshold)
		if (second is None) or (second[7].sum() == 0):
			return None, {'reason': 'no_positive_labels', 'initial_threshold': threshold, 'fallback_threshold': fallback_threshold}
		active = second
	else:
		active = first

	(thr_used, cats, zero_cols, med_cols, drop_info, labeled, X, y) = active

	groups = labeled['CCN'].astype(str).values
	# Retry group splits until test set has at least one positive (or exhaust attempts)
	test_pos = 0
	for attempt in range(10):
		tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state + attempt).split(X, y, groups))
		if y[te].sum() > 0 and y[tr].sum() > 0:
			break
	test_pos = int(y[te].sum())

	Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
	pos_rate = ytr.mean()
	spw = ((1 - pos_rate) / pos_rate) if pos_rate > 0 else 1.0

	clf = xgb.XGBClassifier(
		n_estimators=300, learning_rate=0.05, max_depth=6,
		subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
		eval_metric='aucpr', random_state=random_state, scale_pos_weight=spw
	)
	pipe = Pipeline([('prep', _prep(cats, zero_cols, med_cols)), ('clf', clf)])
	pipe.fit(Xtr, ytr)

	yprob = pipe.predict_proba(Xte)[:, 1]
	if yte.sum() == 0 or yte.sum() == len(yte):
		pr_auc = None
		roc_auc = None
	else:
		pr_auc = float(average_precision_score(yte, yprob))
		roc_auc = float(roc_auc_score(yte, yprob))

	metrics = {
		'threshold_used': thr_used,
		'pr_auc': pr_auc,
		'roc_auc': roc_auc,
		'train_pos_rate': float(pos_rate),
		'test_pos_rate': float(yte.mean()),
		'test_positive_count': test_pos,
		'drop_info': drop_info
	}
	return pipe, metrics

def attach_build_odds(df: pd.DataFrame, threshold: float = 20_000_000):
	pipe, metrics = train_debt_model(df, threshold=threshold)
	out = df.copy()
	# Remove any stale build odds variants
	for c in [c for c in out.columns if c.startswith('Build_Odds')]:
		out = out.drop(columns=c)
	if pipe is None:
		out['Build_Odds'] = np.nan
		metrics['build_odds_attached'] = 0
		return out, metrics
	latest = out.copy()
	latest['Year'] = pd.to_numeric(latest['Year'], errors='coerce')
	latest = latest.sort_values(['CCN','Year'], ascending=[True, False]).drop_duplicates('CCN')
	cats = [c for c in FEATURE_CATS_BASE if c in latest.columns]
	zero_cols = [c for c in FEATURE_ZERO_BASE if c in latest.columns]
	med_cols = [c for c in FEATURE_MED_BASE if c in latest.columns]
	latest_norm = _normalize_features(latest, cats, zero_cols, med_cols)
	use_cols = [c for c in (FEATURE_CATS_BASE + FEATURE_ZERO_BASE + FEATURE_MED_BASE) if c in latest_norm.columns]
	preds = pipe.predict_proba(latest_norm[use_cols])[:, 1]
	latest_map = dict(zip(latest['CCN'], preds))
	# Remove prior Build_Odds if present to avoid suffix merge conflicts
	if 'Build_Odds' in out.columns:
		out = out.drop(columns=['Build_Odds'])
	out['Build_Odds'] = out['CCN'].map(latest_map)
	# Final safety: guarantee only one Build_Odds column
	for c in [c for c in out.columns if c.startswith('Build_Odds') and c != 'Build_Odds']:
		out = out.drop(columns=c)
	metrics['build_odds_attached'] = int(out['Build_Odds'].notna().sum())
	return out, metrics
