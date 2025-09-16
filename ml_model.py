""" xgboost_count_forecast.py

A complete, well-documented training + evaluation + explainability pipeline for predicting exact file counts (non-negative integers) using XGBoost.

Assumptions

You have a CSV file containing daily counts for many (SOURCE_SYSTEM_NM, EXTRACTOR_NAME) combinations. The CSV must include either:

a DATE_VALUE column (YYYY-MM-DD) and a FILE_UP/daily count column OR

LOG_YEAR, LOG_MONTH, LOG_DAY columns (these will be combined into a date) The target column name expected is: FILE_UP (non-negative integer).



Features & modeling choices

Global model across all source+extractor combinations (recommended).

XGBoost using a Poisson objective ("count:poisson") for count targets.

Target encoding (with smoothing) for categorical IDs.

Time features (cyclical encodings), lag features, rolling means.

Group-aware cross-validation (GroupKFold) to evaluate generalization across extractors.

Optional sample reweighting by series frequency to avoid dominant series bias.

SHAP explainability output for feature importance and per-extractor diagnostics.


How to use python xgboost_count_forecast.py --mode train --input data.csv --model_out model.joblib python xgboost_count_forecast.py --mode eval  --input data.csv --model_in model.joblib python xgboost_count_forecast.py --mode predict --input data.csv --model_in model.joblib --forecast_days 14

Output

Model file (joblib), metrics CSV, SHAP plots saved as PNGs, and per-extractor diagnostics CSV.


Dependencies

pandas, numpy, scikit-learn, xgboost, shap, joblib, matplotlib


"""

import argparse import os import json from typing import Tuple, List, Dict

import numpy as np import pandas as pd from sklearn.model_selection import GroupKFold, train_test_split from sklearn.metrics import mean_absolute_error import xgboost as xgb import shap import matplotlib.pyplot as plt import joblib

------------------------- Utility / preprocessing -------------------------

def load_csv(path: str) -> pd.DataFrame: """Loads CSV and normalizes column names to upper-case for robustness.""" df = pd.read_csv(path) df.columns = [c.strip() for c in df.columns] return df

def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame: """Make sure DataFrame has a datetime index column named 'DATE_VALUE'.

Accepts either a pre-existing 'DATE_VALUE' or constructs it from
LOG_YEAR, LOG_MONTH, LOG_DAY if present. The returned DataFrame will have
a column 'DATE_VALUE' of dtype datetime64 and be sorted by date.
"""
if 'DATE_VALUE' in df.columns:
    df['DATE_VALUE'] = pd.to_datetime(df['DATE_VALUE'])
elif {'LOG_YEAR', 'LOG_MONTH', 'LOG_DAY'}.issubset(set(df.columns)):
    df['DATE_VALUE'] = pd.to_datetime(
        df['LOG_YEAR'].astype(str).str.zfill(4) + '-' +
        df['LOG_MONTH'].astype(str).str.zfill(2) + '-' +
        df['LOG_DAY'].astype(str).str.zfill(2)
    )
else:
    raise ValueError("Input CSV must contain DATE_VALUE or LOG_YEAR/LOG_MONTH/LOG_DAY columns")

df = df.sort_values('DATE_VALUE').reset_index(drop=True)
return df

def generate_calendar_features(df: pd.DataFrame) -> pd.DataFrame: """Add calendar and cyclical features and is_weekend flag.

Expects DATE_VALUE column.
"""
df = df.copy()
df['dayofweek'] = df['DATE_VALUE'].dt.dayofweek  # 0=Mon ... 6=Sun
df['dayofmonth'] = df['DATE_VALUE'].dt.day
df['month'] = df['DATE_VALUE'].dt.month
df['weekofyear'] = df['DATE_VALUE'].dt.isocalendar().week.astype(int)
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

# cyclical encodings
df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
df['mon_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
df['mon_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
# week_of_year cyclical (52-week cycle approximation)
df['woy_sin'] = np.sin(2 * np.pi * (df['weekofyear'] - 1) / 52)
df['woy_cos'] = np.cos(2 * np.pi * (df['weekofyear'] - 1) / 52)

return df

def add_lag_and_rolling(df: pd.DataFrame, group_cols: List[str], lags: List[int] = [1, 7], rolling_windows: List[int] = [7]) -> pd.DataFrame: """Create lag features and rolling means per group (series).

This function returns a new DataFrame with added columns:
- file_lag_{k} for each lag k
- rolling_mean_{w} for each rolling window w (shifted so it's past-only)

Parameters
----------
df: must contain 'FILE_UP' and 'DATE_VALUE' and grouping columns.
group_cols: e.g. ['SOURCE_SYSTEM_NM', 'EXTRACTOR_NAME']
"""
out = []
df = df.copy()
df = df.sort_values(group_cols + ['DATE_VALUE'])

def _build_group(g):
    g = g.sort_values('DATE_VALUE').copy()
    for lag in lags:
        g[f'file_lag_{lag}'] = g['FILE_UP'].shift(lag)
    for w in rolling_windows:
        g[f'rolling_mean_{w}'] = g['FILE_UP'].rolling(window=w, min_periods=1).mean().shift(1)
    return g

out = df.groupby(group_cols, group_keys=False).apply(_build_group).reset_index(drop=True)
return out

------------------------- Target encoding (train-fold aware) -------------------------

def fit_target_encoding(train: pd.DataFrame, col: str, target_col: str, min_samples_leaf: int = 50, smoothing: float = 10.0) -> Dict: """Fit a smoothed target encoding mapping on the training fold.

Smoothing formula:
  encoded = (count * mean + smoothing * global_mean) / (count + smoothing)

Returns mapping dict and global mean.
"""
stats = train.groupby(col)[target_col].agg(['count', 'mean'])
global_mean = train[target_col].mean()

# smoothing
counts = stats['count']
means = stats['mean']
smooth = (counts * means + smoothing * global_mean) / (counts + smoothing)

mapping = smooth.to_dict()
return {'mapping': mapping, 'global_mean': global_mean}

def apply_target_encoding(df: pd.DataFrame, col: str, encoding_info: Dict, out_col: str): """Apply mapping, fill missing with global mean.""" mapping = encoding_info['mapping'] gm = encoding_info['global_mean'] df[out_col] = df[col].map(mapping).fillna(gm) return df

------------------------- Modeling & training -------------------------

def build_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, List[str]]: """Return numpy X and feature list (ordered).""" X = df[feature_cols].astype(float) return X.values, list(feature_cols)

def train_with_group_cv(df: pd.DataFrame, features: List[str], target: str = 'FILE_UP', group_cols: List[str] = ['SOURCE_SYSTEM_NM', 'EXTRACTOR_NAME'], n_splits: int = 5, sample_reweight: bool = True, weekend_weight: float = 1.0, xgb_params: dict = None) -> Tuple[xgb.XGBRegressor, Dict]: """Train a global model using GroupKFold and return the best model (by validation MAE)

Steps per fold:
  - compute target encodings on train fold and apply to train/val
  - optional sample reweighting to balance groups
  - train XGBoost (count:poisson by default)
  - evaluate

Returns the best-trained model and a dict with CV metrics and last fold artifacts.
"""
df = df.copy()
# create a single series id for grouping
df['series_id'] = df[group_cols].apply(lambda row: '__'.join(row.values.astype(str)), axis=1)

groups = df['series_id'].values

gkf = GroupKFold(n_splits=n_splits)

if xgb_params is None:
    xgb_params = dict(
        objective='count:poisson',
        tree_method='hist',
        learning_rate=0.05,
        n_estimators=2000,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0
    )

best_mae = float('inf')
best_model = None
fold_results = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df[target], groups)):
    print(f"Starting fold {fold+1}/{n_splits}")
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # target-encode categorical identifiers on the training fold
    # we'll encode both SOURCE_SYSTEM_NM and EXTRACTOR_NAME
    enc_source = fit_target_encoding(train_df, 'SOURCE_SYSTEM_NM', target)
    enc_extr = fit_target_encoding(train_df, 'EXTRACTOR_NAME', target)

    train_df = apply_target_encoding(train_df, 'SOURCE_SYSTEM_NM', enc_source, 'source_te')
    val_df = apply_target_encoding(val_df, 'SOURCE_SYSTEM_NM', enc_source, 'source_te')
    train_df = apply_target_encoding(train_df, 'EXTRACTOR_NAME', enc_extr, 'extractor_te')
    val_df = apply_target_encoding(val_df, 'EXTRACTOR_NAME', enc_extr, 'extractor_te')

    # combine features for this fold
    feat_fold = features + ['source_te', 'extractor_te']

    # sample reweighting by series frequency (optional) to reduce bias towards large series
    if sample_reweight:
        group_counts = train_df['series_id'].value_counts().to_dict()
        # weight = 1 / sqrt(count) so very large series reduce influence
        train_df['sample_weight'] = train_df['series_id'].map(lambda x: 1.0 / np.sqrt(group_counts.get(x, 1)))
        # scale weights to mean 1
        train_df['sample_weight'] *= (len(train_df) / train_df['sample_weight'].sum())
    else:
        train_df['sample_weight'] = 1.0

    # optional weekend upweighting (if needed)
    if weekend_weight != 1.0:
        train_df['sample_weight'] *= np.where(train_df['is_weekend'] == 1, weekend_weight, 1.0)

    X_train = train_df[feat_fold]
    y_train = train_df[target].astype(float)
    w_train = train_df['sample_weight'].values

    X_val = val_df[feat_fold]
    y_val = val_df[target].astype(float)

    model = xgb.XGBRegressor(**xgb_params)
    # early stopping callback
    callbacks = [xgb.callback.EarlyStopping(rounds=150, save_best=True)]

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              sample_weight=w_train,
              verbose=False,
              callbacks=callbacks)

    # predict on validation
    pred_val = model.predict(X_val)
    pred_val_r = np.rint(np.clip(pred_val, 0, None)).astype(int)
    mae = mean_absolute_error(y_val, pred_val_r)
    print(f" Fold {fold+1} MAE: {mae:.4f}")

    fold_results.append({
        'fold': fold+1,
        'model': model,
        'mae': mae,
        'enc_source': enc_source,
        'enc_extr': enc_extr,
        'feat_fold': feat_fold,
        'val_idx': val_idx,
        'X_val': X_val,
        'y_val': y_val,
        'pred_val': pred_val,
    })

    if mae < best_mae:
        best_mae = mae
        best_model = model
        best_artifact = fold_results[-1]

metrics = {
    'best_mae': best_mae,
    'n_splits': n_splits,
    'fold_results': fold_results
}

# return best model and metrics (including encoding info from the best fold)
return best_model, best_artifact, metrics

------------------------- Evaluation utilities -------------------------

def evaluate_exactness(y_true: np.ndarray, y_pred: np.ndarray) -> Dict: y_pred_r = np.rint(np.clip(y_pred, 0, None)).astype(int) mae = np.mean(np.abs(y_true - y_pred_r)) rmse = np.sqrt(np.mean((y_true - y_pred_r)**2)) exact = float(np.mean(y_true == y_pred_r)) within1 = float(np.mean(np.abs(y_true - y_pred_r) <= 1)) return {'MAE': mae, 'RMSE': rmse, 'Exact%': exact, 'Within±1%': within1}

def per_extractor_diagnostics(df: pd.DataFrame, model: xgb.XGBRegressor, feat_cols: List[str]) -> pd.DataFrame: """Return per-extractor metrics and counts to find weak extractors.""" out_rows = [] df = df.copy() # ensure target encodings exist in df (assumes model built with encoders in metadata) # We'll assume provided df already has 'source_te' and 'extractor_te' columns. # If not, user should call apply_target_encoding before this function.

series_keys = df[['SOURCE_SYSTEM_NM', 'EXTRACTOR_NAME']].drop_duplicates()
for _, r in series_keys.iterrows():
    s = r['SOURCE_SYSTEM_NM']
    e = r['EXTRACTOR_NAME']
    sub = df[(df['SOURCE_SYSTEM_NM'] == s) & (df['EXTRACTOR_NAME'] == e)]
    if sub.empty:
        continue
    X_sub = sub[feat_cols]
    y_sub = sub['FILE_UP'].astype(int).values
    preds = np.rint(np.clip(model.predict(X_sub), 0, None)).astype(int)
    metrics = evaluate_exactness(y_sub, preds)
    out_rows.append({
        'SOURCE_SYSTEM_NM': s,
        'EXTRACTOR_NAME': e,
        'n_rows': len(sub),
        **metrics
    })
return pd.DataFrame(out_rows)

------------------------- SHAP explainability -------------------------

def run_shap_analysis(model: xgb.XGBRegressor, X_sample: pd.DataFrame, save_prefix: str = 'shap'): """Compute and save SHAP summary plot and feature importance.

Saves: {save_prefix}_summary.png and {save_prefix}_importance.png
"""
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
out_sum = f"{save_prefix}_summary.png"
plt.savefig(out_sum, dpi=150)
plt.clf()

# built-in gain importance
ax = xgb.plot_importance(model, importance_type='gain', max_num_features=30)
fig = ax.figure
fig.tight_layout()
out_imp = f"{save_prefix}_importance.png"
fig.savefig(out_imp, dpi=150)
plt.close(fig)
print(f"Saved SHAP summary -> {out_sum} and importance -> {out_imp}")

------------------------- Forecasting (recursive) -------------------------

def recursive_forecast_for_series(model: xgb.XGBRegressor, series_df: pd.DataFrame, feature_cols: List[str], forecast_days: int = 14) -> pd.DataFrame: """Recursive forecasting for a single series (source+extractor).

series_df must include the feature columns (including source_te/extractor_te) and
be sorted by DATE_VALUE. Returns a DataFrame indexed by future dates with predicted counts.
"""
history = series_df.sort_values('DATE_VALUE').copy()
last_window = history['FILE_UP'].iloc[-7:].tolist()

rows = []
for i in range(forecast_days):
    next_date = history['DATE_VALUE'].iloc[-1] + pd.Timedelta(days=i+1)
    dow = next_date.dayofweek
    dom = next_date.day
    mon = next_date.month
    woy = next_date.isocalendar().week

    lag_1 = last_window[-1]
    lag_7 = last_window[-7] if len(last_window) >= 7 else last_window[0]
    rolling_mean_7 = float(np.mean(last_window))

    row = {
        'dayofweek': dow,
        'dayofmonth': dom,
        'month': mon,
        'weekofyear': int(woy),
        'is_weekend': int(dow >= 5),
        'dow_sin': np.sin(2*np.pi*dow/7),
        'dow_cos': np.cos(2*np.pi*dow/7),
        'mon_sin': np.sin(2*np.pi*(mon-1)/12),
        'mon_cos': np.cos(2*np.pi*(mon-1)/12),
        'woy_sin': np.sin(2*np.pi*(int(woy)-1)/52),
        'woy_cos': np.cos(2*np.pi*(int(woy)-1)/52),
        'file_lag_1': lag_1,
        'file_lag_7': lag_7,
        'rolling_mean_7': rolling_mean_7,
        # keep source_te/extractor_te constant (use last value from history)
        'source_te': history['source_te'].iloc[-1] if 'source_te' in history.columns else 0.0,
        'extractor_te': history['extractor_te'].iloc[-1] if 'extractor_te' in history.columns else 0.0,
    }

    X_row = pd.DataFrame([row])[feature_cols]
    pred = float(model.predict(X_row)[0])
    pred = max(pred, 0.0)
    pred_rounded = int(np.rint(pred))

    # append to last_window for future lags
    last_window.pop(0)
    last_window.append(pred)

    rows.append({'DATE_VALUE': next_date, 'pred': pred, 'pred_round': pred_rounded})

out = pd.DataFrame(rows).set_index('DATE_VALUE')
return out

------------------------- CLI / Orchestration -------------------------

def main(): p = argparse.ArgumentParser(description='Train / Evaluate / Predict file-count models (XGBoost)') p.add_argument('--mode', choices=['train', 'eval', 'predict'], required=True) p.add_argument('--input', type=str, required=True, help='Path to input CSV') p.add_argument('--model_out', type=str, default='model.joblib') p.add_argument('--model_in', type=str, default='model.joblib') p.add_argument('--forecast_days', type=int, default=14) p.add_argument('--n_splits', type=int, default=5) p.add_argument('--out_dir', type=str, default='artifacts') args = p.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

df_raw = load_csv(args.input)
df = ensure_date_column(df_raw)

# rename target if user used FILE_UP or daily_file_count
if 'FILE_UP' not in df.columns and 'daily_file_count' in df.columns:
    df.rename(columns={'daily_file_count': 'FILE_UP'}, inplace=True)

# enforce required cols
required = {'SOURCE_SYSTEM_NM', 'EXTRACTOR_NAME', 'FILE_UP', 'DATE_VALUE'}
if not required.issubset(set(df.columns)):
    raise ValueError(f"CSV must contain columns: {required}")

# add calendar features
df = generate_calendar_features(df)

# create lags & rolling (per-series)
df = add_lag_and_rolling(df, ['SOURCE_SYSTEM_NM', 'EXTRACTOR_NAME'], lags=[1,7], rolling_windows=[7])

# drop rows with NaNs (first few rows per series won't have lags)
df = df.dropna().reset_index(drop=True)

# standard feature list (without target-encodings)
base_features = [
    'dayofweek','dayofmonth','month','weekofyear','is_weekend',
    'dow_sin','dow_cos','mon_sin','mon_cos','woy_sin','woy_cos',
    'file_lag_1','file_lag_7','rolling_mean_7'
]

if args.mode == 'train':
    print("Starting training with GroupKFold CV...")
    model, best_artifact, metrics = train_with_group_cv(df, base_features, target='FILE_UP', n_splits=args.n_splits)
    # save the best model
    joblib.dump(model, args.model_out)
    print(f"Saved best model -> {args.model_out}")

    # Save fold summary metrics
    fold_summaries = [
        {'fold': r['fold'], 'mae': r['mae'], 'n_val_rows': len(r['y_val'])}
        for r in metrics['fold_results']
    ]
    pd.DataFrame(fold_summaries).to_csv(os.path.join(args.out_dir, 'cv_fold_summary.csv'), index=False)

    # Prepare and save per-extractor diagnostics using the best_artifact encoders
    # Apply best_artifact encoders to the whole dataframe so diagnostics align with model inputs
    enc_source = best_artifact['enc_source']
    enc_extr = best_artifact['enc_extr']
    df_diag = apply_target_encoding(df.copy(), 'SOURCE_SYSTEM_NM', enc_source, 'source_te')
    df_diag = apply_target_encoding(df_diag, 'EXTRACTOR_NAME', enc_extr, 'extractor_te')

    feat_for_preds = best_artifact['feat_fold']
    # evaluate overall
    X_all = df_diag[feat_for_preds]
    preds_all = model.predict(X_all)
    overall_metrics = evaluate_exactness(df_diag['FILE_UP'].values, preds_all)
    with open(os.path.join(args.out_dir, 'model_metrics.json'), 'w') as f:
        json.dump(overall_metrics, f, indent=2)
    print("Overall metrics:", overall_metrics)

    # per-extractor diagnostics
    diag = per_extractor_diagnostics(df_diag, model, feat_for_preds)
    diag.to_csv(os.path.join(args.out_dir, 'per_extractor_diagnostics.csv'), index=False)
    print(f"Saved diagnostics -> {os.path.join(args.out_dir, 'per_extractor_diagnostics.csv')}")

    # SHAP analysis on a sample of validation rows from the best fold
    X_val = best_artifact['X_val']
    run_shap_analysis(model, X_val.sample(min(2000, len(X_val))), save_prefix=os.path.join(args.out_dir, 'shap'))

    print("Training complete. Artifacts saved in:", args.out_dir)

elif args.mode == 'eval':
    print("Loading model and evaluating on provided CSV...")
    model = joblib.load(args.model_in)
    # We need to target-encode using training fold encoders. We don't have them here; approximate by global mappings
    enc_source = fit_target_encoding(df, 'SOURCE_SYSTEM_NM', 'FILE_UP')
    enc_extr = fit_target_encoding(df, 'EXTRACTOR_NAME', 'FILE_UP')
    df = apply_target_encoding(df, 'SOURCE_SYSTEM_NM', enc_source, 'source_te')
    df = apply_target_encoding(df, 'EXTRACTOR_NAME', enc_extr, 'extractor_te')

    # generate predictions and metrics
    feat_cols = base_features + ['source_te', 'extractor_te']
    X = df[feat_cols]
    preds = model.predict(X)
    metrics = evaluate_exactness(df['FILE_UP'].values, preds)
    print("Evaluation metrics (using rounded predictions):", metrics)

    # per-extractor diagnostics
    diag = per_extractor_diagnostics(df, model, feat_cols)
    diag.to_csv(os.path.join(args.out_dir, 'per_extractor_diagnostics_eval.csv'), index=False)
    print(f"Saved diagnostics -> {os.path.join(args.out_dir, 'per_extractor_diagnostics_eval.csv')}")

elif args.mode == 'predict':
    print("Loading model and producing recursive forecasts per series...")
    model = joblib.load(args.model_in)
    # Fit quick encodings from entire CSV so predictions have source_te/extractor_te
    enc_source = fit_target_encoding(df, 'SOURCE_SYSTEM_NM', 'FILE_UP')
    enc_extr = fit_target_encoding(df, 'EXTRACTOR_NAME', 'FILE_UP')
    df = apply_target_encoding(df, 'SOURCE_SYSTEM_NM', enc_source, 'source_te')
    df = apply_target_encoding(df, 'EXTRACTOR_NAME', enc_extr, 'extractor_te')

    feat_cols = base_features + ['source_te', 'extractor_te']

    # for each series produce forecast and save
    forecasts = []
    by_series = df.groupby(['SOURCE_SYSTEM_NM', 'EXTRACTOR_NAME'], group_keys=False)
    for (s, e), g in by_series:
        g_sorted = g.sort_values('DATE_VALUE').reset_index(drop=True)
        # ensure features exist
        if set(feat_cols).issubset(set(g_sorted.columns)):
            f_df = recursive_forecast_for_series(model, g_sorted, feat_cols, forecast_days=args.forecast_days)
            f_df['SOURCE_SYSTEM_NM'] = s
            f_df['EXTRACTOR_NAME'] = e
            forecasts.append(f_df.reset_index())

    if len(forecasts) == 0:
        print("No forecasts produced (check if features exist).")
        return

    all_fore = pd.concat(forecasts, axis=0, ignore_index=True)
    out_csv = os.path.join(args.out_dir, f'forecasts_{args.forecast_days}d.csv')
    all_fore.to_csv(out_csv, index=False)
    print(f"Saved forecasts -> {out_csv}")

else:
    raise ValueError('Unknown mode')

if name == 'main': main()

