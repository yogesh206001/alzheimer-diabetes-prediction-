"""
===========================================================================
  UNIFIED PIPELINE: Diabetes + Alzheimer (DARWIN) Data Fusion & Analysis
===========================================================================
  Author      : Research Pipeline
  Purpose     : Fuse clinical diabetes data with DARWIN handwriting-based
                cognitive features, engineer a medically realistic
                alzheimer_risk_score, validate data quality, train and
                compare ML models, and explain predictions with SHAP.
  Dataset 1   : Pima Indians Diabetes Dataset (clinical features)
  Dataset 2   : DARWIN Handwriting Dataset (452 cognitive features)
  Output Files:
      - advanced_fused_dataset.csv   : Final normalized fused dataset
      - correlation_matrix.csv       : Full Pearson correlation matrix
      - feature_importance.csv       : Random Forest feature rankings
      - shap_summary.png             : SHAP beeswarm summary plot
      - validation_report.txt        : Full data validation log
===========================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 ─ Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

matplotlib.use('Agg')          # Non-interactive backend (saves files, no GUI)
warnings.simplefilter('ignore')
np.random.seed(42)             # Global seed for full reproducibility

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 ─ Load and Reduce DARWIN Alzheimer Dataset
#
# The DARWIN dataset records 25 handwriting tasks per patient, yielding 452
# raw features. We aggregate each key metric across all 25 tasks to obtain 7
# compact, interpretable, and medically meaningful features.
#
# Cognitive Decline Rationale (Alzheimer's):
#   - mean_total_time    : Slower processing speed indicates neural decline
#   - mean_pressure_mean : Loss of muscle tone/regulation affects pen pressure
#   - mean_pressure_var  : Erratic pressure signals unstable motor control
#   - mean_speed_on_paper: Reduced speed reflects impaired motor execution
#   - mean_gmrt_on_paper : Longer reaction times indicate neural delay
#   - mean_jerk_on_paper : More jerkiness = loss of smooth motor coordination
#   - std_total_time     : High variance across tasks = cognitive inconsistency
# ─────────────────────────────────────────────────────────────────────────────
def load_darwin(darwin_path: str) -> pd.DataFrame:
    print("[1/7] Loading and reducing DARWIN dataset...")
    df = pd.read_csv(darwin_path)

    task_cols = {
        'mean_total_time'     : [f'total_time{i}'          for i in range(1, 26)],
        'mean_pressure_mean'  : [f'pressure_mean{i}'       for i in range(1, 26)],
        'mean_pressure_var'   : [f'pressure_var{i}'        for i in range(1, 26)],
        'mean_speed_on_paper' : [f'mean_speed_on_paper{i}' for i in range(1, 26)],
        'mean_gmrt_on_paper'  : [f'gmrt_on_paper{i}'       for i in range(1, 26)],
        'mean_jerk_on_paper'  : [f'mean_jerk_on_paper{i}'  for i in range(1, 26)],
    }

    reduced = pd.DataFrame()
    for feat, cols in task_cols.items():
        reduced[feat] = df[cols].mean(axis=1)

    # Global variability: std of total_time across all 25 tasks
    reduced['std_total_time'] = df[[f'total_time{i}' for i in range(1, 26)]].std(axis=1)

    # Winsorise at 1st–99th percentile to remove extreme sensor artefacts
    for col in reduced.columns:
        lo, hi = reduced[col].quantile([0.01, 0.99])
        reduced[col] = np.clip(reduced[col], lo, hi)

    print(f"    DARWIN reduced: {df.shape[1]} -> {reduced.shape[1]} features | {reduced.shape[0]} patients")
    return reduced


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 ─ Load, Clean, and Validate Pima Indians Diabetes Dataset
#
# Clinical Cleaning Rules Applied:
#   - Glucose, BloodPressure, SkinThickness, Insulin, BMI: zeros are
#     physiologically impossible → replaced with NaN → median-imputed.
#   - Hard clinical bounds: Glucose ∈ [40, 400], BMI ∈ [15, 65],
#     Age ∈ [21, 100].
# ─────────────────────────────────────────────────────────────────────────────
def load_diabetes() -> pd.DataFrame:
    print("[2/7] Fetching and cleaning Pima Indians Diabetes dataset...")
    url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
           "pima-indians-diabetes.data.csv")
    cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(url, header=None, names=cols)

    # Replace physiologically impossible zeroes with NaN, then median-impute
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[zero_cols] = df[zero_cols].replace(0, np.nan)
    df.fillna(df.median(numeric_only=True), inplace=True)

    before = len(df)
    df = df[(df['Glucose']  >= 40)  & (df['Glucose']  <= 400)]
    df = df[(df['BMI']      >= 15)  & (df['BMI']      <= 65)]
    df = df[(df['Age']      >= 21)  & (df['Age']      <= 100)]
    print(f"    Diabetes rows after outlier removal: {before} -> {len(df)}")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 ─ Synthetic Data Fusion (Column-wise Bootstrapped Merge)
#
# Since the two datasets contain DIFFERENT patients, we cannot row-align them
# by identity.  Instead we independently bootstrap N=1000 samples from each
# dataset and concatenate column-wise.  This preserves the statistical
# distributions of both datasets while allowing us to study combined features.
# ─────────────────────────────────────────────────────────────────────────────
def fuse_datasets(diab_df: pd.DataFrame, alz_df: pd.DataFrame,
                  n: int = 1000) -> pd.DataFrame:
    print(f"[3/7] Performing synthetic data fusion (N={n})...")
    diab_s = diab_df.sample(n=n, replace=True, random_state=43).reset_index(drop=True)
    alz_s  = alz_df.sample( n=n, replace=True, random_state=44).reset_index(drop=True)
    fused  = pd.concat([diab_s, alz_s], axis=1)
    print(f"    Fused shape: {fused.shape}")
    return fused


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 ─ Non-Linear alzheimer_risk_score Engineering
#
# Formula (applied on raw un-normalised values so thresholds are meaningful):
#
#   Age effect   = Age + (Age - 60)^1.5   [if Age > 60]
#   Gluc effect  = Gluc + (Gluc - 140)^1.4 [if Gluc > 140]
#   DM factor    = Outcome  ∈ {0, 1}
#   Cog composite= 0.35·mean_time + 0.25·mean_jerk + 0.20·std_time
#                  + 0.20·(1 − mean_speed)    (all MinMax-scaled)
#
#   Raw score    = 0.30·age_comp + 0.20·gluc_comp
#                + 0.15·dm_comp  + 0.35·cog_comp
#   Final score  = MinMaxScaler(raw score + N(0, 0.04))  ∈ [0, 1]
#
# Medical rationale for non-linearity:
#   • Age: Alzheimer's incidence roughly doubles every 5 years after 65.
#   • Glucose: Prediabetes (>100) poses moderate risk; overt hyperglycaemia
#     (>140) drives hippocampal inflammation and tau phosphorylation.
# ─────────────────────────────────────────────────────────────────────────────
def engineer_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    print("[4/7] Engineering non-linear alzheimer_risk_score...")
    scaler = MinMaxScaler()

    # Age component with exponential spike after 60
    age_raw   = df['Age'].values
    age_eff   = age_raw + np.where(age_raw > 60, (age_raw - 60) ** 1.5, 0.0)
    age_comp  = scaler.fit_transform(age_eff.reshape(-1, 1)).flatten()

    # Glucose component with exponential spike after 140
    gluc_raw  = df['Glucose'].values
    gluc_eff  = gluc_raw + np.where(gluc_raw > 140, (gluc_raw - 140) ** 1.4, 0.0)
    gluc_comp = scaler.fit_transform(gluc_eff.reshape(-1, 1)).flatten()

    # Diabetes diagnosis binary flag
    dm_comp   = df['Outcome'].values.astype(float)

    # Cognitive composite (all sub-features normalised before weighting)
    cog_scaled = pd.DataFrame(
        scaler.fit_transform(df[['mean_total_time', 'mean_jerk_on_paper',
                                  'std_total_time',  'mean_speed_on_paper']]),
        columns=['t', 'j', 's', 'v']
    )
    cog_comp = (cog_scaled['t'] * 0.35 +
                cog_scaled['j'] * 0.25 +
                cog_scaled['s'] * 0.20 +
                (1.0 - cog_scaled['v']) * 0.20)
    cog_comp = scaler.fit_transform(cog_comp.values.reshape(-1, 1)).flatten()

    # Weighted combination + controlled biological noise (σ = 0.04)
    raw = (age_comp * 0.30 + gluc_comp * 0.20 +
           dm_comp  * 0.15 + cog_comp  * 0.35)
    raw += np.random.normal(scale=0.04, size=len(df))

    # Strictly bound to [0, 1]
    df['alzheimer_risk_score'] = scaler.fit_transform(raw.reshape(-1, 1)).flatten()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 ─ Normalize All Features to [0, 1]
# ─────────────────────────────────────────────────────────────────────────────
def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print("[5/7] Normalizing all features to [0, 1]...")
    scaler    = MinMaxScaler()
    feat_cols = [c for c in df.columns if c != 'alzheimer_risk_score']
    df[feat_cols] = scaler.fit_transform(df[feat_cols])

    missing = df.isnull().sum().sum()
    print(f"    Missing values after normalization: {missing}")
    assert missing == 0, "ERROR: Missing values detected!"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 ─ Correlation Analysis
# ─────────────────────────────────────────────────────────────────────────────
def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    print("[6/7] Computing correlation matrix...")
    corr = df.corr()
    corr.to_csv("correlation_matrix.csv")

    risk_corr = corr['alzheimer_risk_score'].drop('alzheimer_risk_score').sort_values(ascending=False)
    print("\n    === Correlation with alzheimer_risk_score ===")
    print(risk_corr.to_string())
    return corr


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 ─ Model Training, Evaluation, Feature Importance, and SHAP
# ─────────────────────────────────────────────────────────────────────────────
def train_and_explain(df: pd.DataFrame) -> None:
    print("\n[7/7] Training models and running SHAP...")

    X = df.drop('alzheimer_risk_score', axis=1)
    y = df['alzheimer_risk_score']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # ── 7a. Model Comparison ──────────────────────────────────────────────────
    models = {
        "Linear Regression"      : LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
                                        n_estimators=100,
                                        max_depth=7,
                                        random_state=42),
    }

    results = []
    trained = {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        results.append({
            "Model"   : name,
            "R2 Score": round(r2_score(y_test, y_pred), 4),
            "MSE"     : round(mean_squared_error(y_test, y_pred), 6),
        })
        trained[name] = mdl

    res_df = pd.DataFrame(results).sort_values("R2 Score", ascending=False)
    print("\n    === Model Performance Comparison ===")
    print(res_df.to_string(index=False))

    best_name = res_df.iloc[0]['Model']
    print(f"\n    Best model: {best_name}")

    # ── 7b. Random Forest Feature Importance ─────────────────────────────────
    rf_model = trained["Random Forest Regressor"]
    imp_df = pd.DataFrame({
        'Feature'   : X.columns,
        'Importance': rf_model.feature_importances_,
    }).sort_values('Importance', ascending=False)
    imp_df.to_csv("feature_importance.csv", index=False)
    print("\n    === Random Forest Feature Importances ===")
    print(imp_df.to_string(index=False))

    # ── 7c. SHAP Analysis on the Best (Linear Regression) Model ──────────────
    lr_model = trained["Linear Regression"]
    explainer   = shap.LinearExplainer(lr_model, X_train)
    shap_values = explainer.shap_values(X)

    # Beeswarm summary plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=300, bbox_inches='tight')
    print("\n    SHAP summary plot saved -> shap_summary.png")

    # SHAP importance table
    shap_imp = pd.DataFrame({
        'Feature'      : X.columns,
        'Mean_Abs_SHAP': np.abs(shap_values).mean(axis=0),
    }).sort_values('Mean_Abs_SHAP', ascending=False)
    print("\n    === SHAP Feature Importance Ranking ===")
    print(shap_imp.to_string(index=False))

    # ── 7d. Validation Report ─────────────────────────────────────────────────
    report = f"""
=============================================================
  FULL PIPELINE VALIDATION REPORT
=============================================================

1. DATA CLEANING
   - DARWIN: Winsorized at [1%%, 99%%] per feature (452 -> 7 features)
   - Diabetes: Zero-imputation with median; clinical bounds enforced
   - Post-fusion missing values: 0

2. ALZHEIMER RISK SCORE FORMULA
   age_comp   = MinMax( Age  + (Age  - 60 )^1.5  if Age  > 60  else 0 )
   gluc_comp  = MinMax( Gluc + (Gluc - 140)^1.4  if Gluc > 140 else 0 )
   dm_comp    = Outcome in [0, 1]
   cog_comp   = MinMax( 0.35*mean_time + 0.25*mean_jerk
                       + 0.20*std_time  + 0.20*(1 - mean_speed) )
   raw        = 0.30*age + 0.20*gluc + 0.15*dm + 0.35*cog + N(0,0.04)
   FINAL      = MinMaxScaler(raw)  in [0, 1]

3. MODEL COMPARISON (80/20 Train/Test Split)
{res_df.to_string(index=False)}

4. RANDOM FOREST FEATURE IMPORTANCE
{imp_df.to_string(index=False)}

5. SHAP IMPORTANCE (Linear Regression)
{shap_imp.to_string(index=False)}

=============================================================
"""
    with open("validation_report.txt", "w") as f:
        f.write(report)
    print("\n    Full validation report saved -> validation_report.txt")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    darwin_path = "DARWIN.csv"
    if not os.path.exists(darwin_path):
        raise FileNotFoundError(f"{darwin_path} not found.")

    print("=" * 65)
    print("  Diabetes + Alzheimer Unified Analysis Pipeline")
    print("=" * 65)

    # Load
    alz_df  = load_darwin(darwin_path)
    diab_df = load_diabetes()

    # Fuse
    fused   = fuse_datasets(diab_df, alz_df, n=1000)

    # Engineer risk score
    fused   = engineer_risk_score(fused)

    # Normalize
    fused   = normalize_dataset(fused)

    # Save master CSV
    out = "advanced_fused_dataset.csv"
    fused.to_csv(out, index=False)
    print(f"\n    Final dataset saved -> {out}  {fused.shape}")

    # Correlations
    correlation_analysis(fused)

    # Train, evaluate, feature importance, SHAP
    train_and_explain(fused)

    print("\n" + "=" * 65)
    print("  Pipeline complete. All outputs saved.")
    print("=" * 65)


if __name__ == "__main__":
    main()
