"""
export_final_model.py
=====================
Trains XGBoost on the full advanced_fused_dataset.csv
and exports:
  - final_model.pkl   : trained XGBoostRegressor
  - scaler.pkl        : fitted MinMaxScaler  (raw -> [0,1])
  - feature_meta.pkl  : feature_order list + feature_defaults dict

Run once before starting the Django server:
    python export_final_model.py
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

# ── constants ────────────────────────────────────────────────────────────────
FEATURE_ORDER = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome',
    'mean_total_time', 'mean_pressure_mean', 'mean_pressure_var',
    'mean_speed_on_paper', 'mean_gmrt_on_paper', 'mean_jerk_on_paper',
    'std_total_time'
]
TARGET = 'alzheimer_risk_score'


def build_raw_dataset():
    """
    Reconstruct a raw (un-scaled) version of the fused dataset so we can
    fit a proper MinMaxScaler.  The advanced_fused_dataset.csv values are
    already in [0,1], so we read DARWIN directly for the handwriting side
    and re-build the diabetes side from the CSV (the normalised values map
    back to raw via known column ranges).
    """
    print("  [raw] Reading DARWIN.csv ...")
    darwin = pd.read_csv("DARWIN.csv")

    task_cols = {
        'mean_total_time'     : [f'total_time{i}'          for i in range(1, 26)],
        'mean_pressure_mean'  : [f'pressure_mean{i}'       for i in range(1, 26)],
        'mean_pressure_var'   : [f'pressure_var{i}'        for i in range(1, 26)],
        'mean_speed_on_paper' : [f'mean_speed_on_paper{i}' for i in range(1, 26)],
        'mean_gmrt_on_paper'  : [f'gmrt_on_paper{i}'       for i in range(1, 26)],
        'mean_jerk_on_paper'  : [f'mean_jerk_on_paper{i}'  for i in range(1, 26)],
    }
    alz = pd.DataFrame()
    for feat, cols in task_cols.items():
        alz[feat] = darwin[cols].mean(axis=1)
    alz['std_total_time'] = darwin[[f'total_time{i}' for i in range(1, 26)]].std(axis=1)

    # clip outliers
    for col in alz.columns:
        lo, hi = alz[col].quantile([0.01, 0.99])
        alz[col] = np.clip(alz[col], lo, hi)

    print("  [raw] Simulating diabetes side from Pima medians ...")
    # realistic raw ranges for PIMA features
    pima_ranges = {
        'Pregnancies'            : (0,  17),
        'Glucose'                : (44, 199),
        'BloodPressure'          : (24, 110),
        'SkinThickness'          : (7,  63),
        'Insulin'                : (14, 846),
        'BMI'                    : (18.2, 67.1),
        'DiabetesPedigreeFunction': (0.078, 2.42),
        'Age'                    : (21, 81),
        'Outcome'                : (0, 1),
    }

    # Load the scaled dataset to get Outcome column
    scaled_df = pd.read_csv("advanced_fused_dataset.csv")

    n = 1000
    np.random.seed(42)
    diab_raw = pd.DataFrame()
    for col, (lo, hi) in pima_ranges.items():
        if col == 'Outcome':
            diab_raw[col] = scaled_df['Outcome'].sample(n=n, replace=True, random_state=42).values
        else:
            diab_raw[col] = np.random.uniform(lo, hi, n)

    alz_s = alz.sample(n=n, replace=True, random_state=44).reset_index(drop=True)
    raw_df = pd.concat([diab_raw.reset_index(drop=True), alz_s], axis=1)
    raw_df = raw_df[FEATURE_ORDER]

    defaults = {col: float(raw_df[col].median()) for col in FEATURE_ORDER}
    return raw_df, defaults


def main():
    SEP = "=" * 60
    print(SEP)
    print("  EXPORT FINAL MODEL  (XGBoost)")
    print(SEP)

    # 1. Build raw dataset for scaler
    print("\n[1] Building raw dataset for scaler ...")
    raw_df, defaults = build_raw_dataset()

    # 2. Fit scaler on raw features
    print("[2] Fitting MinMaxScaler ...")
    scaler = MinMaxScaler()
    scaler.fit(raw_df)

    # 3. Load scaled dataset + target for model training
    print("[3] Loading advanced_fused_dataset.csv ...")
    final_df = pd.read_csv("advanced_fused_dataset.csv")
    X = final_df[FEATURE_ORDER].values
    y = final_df[TARGET].values
    print(f"     Samples: {len(X)} | Features: {X.shape[1]}")

    # 4. Train XGBoost on full dataset
    print("[4] Training XGBoost Regressor (full dataset) ...")
    xgb = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    xgb.fit(X, y)

    # Quick sanity check
    from sklearn.metrics import r2_score
    preds = np.clip(xgb.predict(X), 0, 1)
    print(f"     Train R2 (full data): {r2_score(y, preds):.4f}")

    # 5. Feature importances for explainability
    importances = dict(zip(FEATURE_ORDER, xgb.feature_importances_.tolist()))

    # 6. Save artefacts
    print("[5] Saving artefacts ...")

    joblib.dump(xgb,    "final_model.pkl")
    print("     -> final_model.pkl")

    joblib.dump(scaler, "scaler.pkl")
    print("     -> scaler.pkl")

    meta = {
        "feature_order"    : FEATURE_ORDER,
        "feature_defaults" : defaults,
        "feature_importances": importances,
    }
    joblib.dump(meta, "feature_meta.pkl")
    print("     -> feature_meta.pkl")

    # 7. Also update alzheimer_model.pkl for backward compatibility
    compat = {
        'model'           : xgb,
        'scaler'          : scaler,
        'feature_order'   : FEATURE_ORDER,
        'feature_defaults': defaults,
    }
    joblib.dump(compat, "alzheimer_model.pkl")
    print("     -> alzheimer_model.pkl  (backward-compat)")

    print()
    print(SEP)
    print("  DONE.  Ready for Django deployment.")
    print(SEP)
    print("\nTop Feature Importances (XGBoost):")
    for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:8]:
        bar = "#" * int(imp * 40)
        print(f"  {feat:<32} {imp:.4f}  {bar}")


if __name__ == "__main__":
    main()
