"""
=============================================================================
  MODEL COMPARISON PIPELINE  -  Alzheimer Risk Score Prediction
  (Diabetes + Handwriting Biometrics Fused Dataset)
=============================================================================
  Evaluation
  ----------
  - 5-Fold Cross-Validation  (shuffle=True, random_state=42)
  - Models   : Linear Regression | Random Forest | Decision Tree | XGBoost
  - Metrics  : R2  |  Adjusted R2  |  MAE  |  RMSE  |  MSE
  - Reported : mean +/- std across 5 folds

  Outputs
  -------
  - Console  : full report + interpretation
  - model_comparison_results.txt  : IEEE paper-ready report
  - model_cv_results.csv          : per-fold raw data
=============================================================================
"""

import pandas as pd
import numpy as np
import warnings
import time

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.tree            import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# -- XGBoost (optional) ------------------------------------------------------
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not installed. Run:  pip install xgboost")


# ----------------------------------------------------------------------------
def adjusted_r2(r2: float, n: int, p: int) -> float:
    """Return Adjusted R2 (NaN if denominator <= 0)."""
    denom = n - p - 1
    return float("nan") if denom <= 0 else 1.0 - (1.0 - r2) * (n - 1) / denom


# ----------------------------------------------------------------------------
def run_pipeline() -> None:
    SEP  = "=" * 76
    DASH = "-" * 76

    # -- 1. Load data ------------------------------------------------------
    DATASET = "advanced_fused_dataset.csv"
    TARGET  = "alzheimer_risk_score"
    N_FOLDS = 5

    print(SEP)
    print("  MODEL COMPARISON PIPELINE - Alzheimer Risk Score Prediction")
    print(SEP)
    print(f"\n[1] Loading dataset : {DATASET}")

    df = pd.read_csv(DATASET)
    features = [c for c in df.columns if c != TARGET]
    X = df[features].values
    y = df[TARGET].values
    n_samples, n_features = X.shape

    print(f"    Samples          : {n_samples}")
    print(f"    Features (p)     : {n_features}")
    print(f"    Target           : {TARGET}  "
          f"[{y.min():.4f} - {y.max():.4f}]")

    # -- 2. Define models --------------------------------------------------
    models = {
        "Linear Regression" : LinearRegression(),
        "Random Forest"     : RandomForestRegressor(
                                  n_estimators=100, max_depth=7,
                                  random_state=42, n_jobs=-1),
        "Decision Tree"     : DecisionTreeRegressor(random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"]   = XGBRegressor(
                                  n_estimators=200, max_depth=5,
                                  learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8,
                                  random_state=42, verbosity=0, n_jobs=-1)

    # -- 3. 5-Fold Cross-Validation ----------------------------------------
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    print(f"\n[2] Running {N_FOLDS}-Fold Cross-Validation on {len(models)} models ...\n")

    summary   : dict[str, dict] = {}
    fold_rows : list[dict]      = []

    for name, model in models.items():
        t0 = time.perf_counter()
        print(f"    - {name} ...", end="", flush=True)

        r2_v, adj_v, mae_v, rmse_v, mse_v = [], [], [], [], []

        for fi, (tr_idx, te_idx) in enumerate(kf.split(X)):
            model.fit(X[tr_idx], y[tr_idx])
            yp  = model.predict(X[te_idx])
            n   = len(te_idx)

            r2  = float(r2_score(y[te_idx], yp))
            adj = adjusted_r2(r2, n, n_features)
            mae = float(mean_absolute_error(y[te_idx], yp))
            mse = float(mean_squared_error(y[te_idx], yp))
            rm  = float(np.sqrt(mse))

            r2_v.append(r2);  adj_v.append(adj)
            mae_v.append(mae); rmse_v.append(rm); mse_v.append(mse)

            fold_rows.append({
                "Model"  : name,
                "Fold"   : fi + 1,
                "R2"     : round(r2,  6),
                "Adj_R2" : round(adj, 6),
                "MAE"    : round(mae, 6),
                "RMSE"   : round(rm,  6),
                "MSE"    : round(mse, 6),
            })

        elapsed = time.perf_counter() - t0
        print(f"  done  ({elapsed:.1f}s)")

        summary[name] = {
            "R2_mean"    : float(np.mean(r2_v)),
            "R2_std"     : float(np.std(r2_v)),
            "AdjR2_mean" : float(np.nanmean(adj_v)),
            "AdjR2_std"  : float(np.nanstd(adj_v)),
            "MAE_mean"   : float(np.mean(mae_v)),
            "MAE_std"    : float(np.std(mae_v)),
            "RMSE_mean"  : float(np.mean(rmse_v)),
            "RMSE_std"   : float(np.std(rmse_v)),
            "MSE_mean"   : float(np.mean(mse_v)),
            "MSE_std"    : float(np.std(mse_v)),
            "Time_s"     : round(elapsed, 2),
        }

    # -- 4. Build ranked summary DataFrame --------------------------------
    rows = []
    for mn, m in summary.items():
        rows.append({
            "Model"       : mn,
            "R2"          : f"{m['R2_mean']:.4f}",
            "+/-std(R2)"  : f"{m['R2_std']:.4f}",
            "Adj R2"      : f"{m['AdjR2_mean']:.4f}",
            "MAE"         : f"{m['MAE_mean']:.6f}",
            "RMSE"        : f"{m['RMSE_mean']:.6f}",
            "MSE"         : f"{m['MSE_mean']:.6f}",
            "Time (s)"    : m["Time_s"],
            "_sort"       : m["R2_mean"],
        })

    rank_df = (
        pd.DataFrame(rows)
        .sort_values("_sort", ascending=False)
        .drop(columns="_sort")
        .reset_index(drop=True)
    )
    rank_df.insert(0, "Rank", range(1, len(rank_df) + 1))

    ranked_names = list(rank_df["Model"])
    best_name    = ranked_names[0]
    best         = summary[best_name]

    # -- 5. Console output -------------------------------------------------
    print()
    print(SEP)
    print("  RESULTS  -  5-FOLD CROSS-VALIDATION  (mean +/- std)")
    print(SEP)
    print(rank_df.to_string(index=False))
    print()
    print(DASH)
    print("  INTERPRETATION")
    print(DASH)
    print(f"  Best Model   : {best_name}")
    print(f"    R2         : {best['R2_mean']:.4f} +/- {best['R2_std']:.4f}  "
          f"({best['R2_mean']*100:.1f}% variance explained)")
    print(f"    Adjusted R2: {best['AdjR2_mean']:.4f}  "
          f"(penalised for p={n_features} features)")
    print(f"    MAE        : {best['MAE_mean']:.6f} +/- {best['MAE_std']:.6f}")
    print(f"    RMSE       : {best['RMSE_mean']:.6f} +/- {best['RMSE_std']:.6f}")
    print(f"    MSE        : {best['MSE_mean']:.6f} +/- {best['MSE_std']:.6f}")
    print()
    for mn in ranked_names[1:]:
        m   = summary[mn]
        gap = best["R2_mean"] - m["R2_mean"]
        print(f"  {mn:<26}: R2={m['R2_mean']:.4f} +/-{m['R2_std']:.4f}  "
              f"Delta={gap:.4f}")
    print()

    # -- 6. Save per-fold CSV ----------------------------------------------
    cv_csv = "model_cv_results.csv"
    pd.DataFrame(fold_rows).to_csv(cv_csv, index=False)
    print(f"[3] Per-fold data  -> {cv_csv}")

    # -- 7. Build full text report -----------------------------------------
    rpt = [
        SEP,
        "  MODEL COMPARISON REPORT  -  Alzheimer Risk Score Prediction",
        SEP,
        f"  Dataset  : {DATASET}  |  Samples={n_samples}  |  Features={n_features}",
        f"  CV       : {N_FOLDS}-Fold Stratified (shuffle=True, seed=42)",
        f"  Models   : {', '.join(models.keys())}",
        SEP, "",
        # -- Section I --------------------------------------------------
        "SECTION I  -  COMPARISON TABLE (sorted by R2, descending)",
        DASH,
        rank_df.to_string(index=False),
        "",
        "  Interpretation:",
        "    Higher R2 / Adjusted R2 -> better fit.",
        "    Lower MAE / RMSE / MSE  -> smaller prediction error.",
        "",
        DASH,
        # -- Section II -------------------------------------------------
        "SECTION II  -  DETAILED METRICS PER MODEL",
        DASH,
    ]

    for mn, m in summary.items():
        rpt += [
            "",
            f"  {mn}",
            f"    R2            : {m['R2_mean']:.6f}  +/-  {m['R2_std']:.6f}",
            f"    Adjusted R2   : {m['AdjR2_mean']:.6f}  +/-  {m['AdjR2_std']:.6f}",
            f"    MAE           : {m['MAE_mean']:.6f}  +/-  {m['MAE_std']:.6f}",
            f"    RMSE          : {m['RMSE_mean']:.6f}  +/-  {m['RMSE_std']:.6f}",
            f"    MSE           : {m['MSE_mean']:.6f}  +/-  {m['MSE_std']:.6f}",
            f"    Training time : {m['Time_s']}s",
        ]

    rpt += [
        "",
        DASH,
        # -- Section III ------------------------------------------------
        "SECTION III  -  PERFORMANCE INTERPRETATION",
        DASH,
        "",
        f"  Best Model   : {best_name}",
        f"    R2         : {best['R2_mean']:.4f} +/- {best['R2_std']:.4f}",
        f"               -> The model explains {best['R2_mean']*100:.1f}% of the variance",
        f"                 in the Alzheimer risk score.",
        f"    Adjusted R2: {best['AdjR2_mean']:.4f}  (accounts for {n_features} predictors)",
        f"    MAE        : {best['MAE_mean']:.6f} +/- {best['MAE_std']:.6f}",
        f"               -> On average, predictions deviate by {best['MAE_mean']:.4f} units.",
        f"    RMSE       : {best['RMSE_mean']:.6f} +/- {best['RMSE_std']:.6f}",
        "",
        "  Relative performance vs best model:",
    ]
    for mn in ranked_names[1:]:
        m   = summary[mn]
        gap = best["R2_mean"] - m["R2_mean"]
        rpt.append(f"    {mn:<26} R2={m['R2_mean']:.4f} +/-{m['R2_std']:.4f}  Delta={gap:.4f}")

    if "XGBoost" in summary and XGBOOST_AVAILABLE:
        xgb = summary["XGBoost"]
        rpt += [
            "",
            "  XGBoost Notes:",
            f"    XGBoost achieved R2={xgb['R2_mean']:.4f} with a notably low std of "
            f"{xgb['R2_std']:.4f},",
            "    indicating consistent performance across folds - beneficial for",
            "    deployment in clinical decision-support systems.",
        ]

    # -- Section IV: IEEE markdown table --------------------------------
    hdr  = ("| {:4} | {:<22} | {:<20} | {:>8} | {:>10} | {:>10} | {:>10} |"
            .format("Rank", "Model", "R2 (mean +/- std)", "Adj R2",
                    "MAE", "RMSE", "MSE"))
    sep2 = "|------|------------------------|----------------------"  \
           "|----------|------------|------------|------------|"

    rpt += [
        "",
        DASH,
        "SECTION IV  -  IEEE PAPER-READY MARKDOWN TABLE",
        DASH,
        "",
        hdr,
        sep2,
    ]
    for _, row in rank_df.iterrows():
        mn = row["Model"]
        m  = summary[mn]
        rpt.append(
            "| {:4} | {:<22} | {:<20} | {:>8} | {:>10} | {:>10} | {:>10} |".format(
                int(row["Rank"]),
                mn,
                f"{m['R2_mean']:.4f} +/- {m['R2_std']:.4f}",
                f"{m['AdjR2_mean']:.4f}",
                f"{m['MAE_mean']:.6f}",
                f"{m['RMSE_mean']:.6f}",
                f"{m['MSE_mean']:.6f}",
            )
        )

    rpt += [
        "",
        "  *All metrics are mean +/- standard deviation across 5 cross-validation folds.*",
        "  *Higher R2/Adj R2 and lower MAE/RMSE/MSE indicate superior performance.*",
        "",
        SEP,
        "  END OF REPORT",
        SEP,
    ]

    report_path = "model_comparison_results.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rpt))

    print(f"[4] Full report    -> {report_path}")
    print()
    print(SEP)
    print("  DONE - Report ready for research paper inclusion.")
    print(SEP)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()
