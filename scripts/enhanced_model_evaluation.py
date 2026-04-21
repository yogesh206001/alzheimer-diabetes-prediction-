"""
Enhanced Model Evaluation Pipeline
====================================
5-Fold Cross-Validation for:
  - Linear Regression
  - Random Forest Regressor
  - Decision Tree Regressor
  - XGBoost Regressor

Metrics (mean +/- std across 5 folds):
  R2, Adjusted R2, MAE, RMSE, MSE

Outputs:
  enhanced_model_results.txt  -- report for IEEE paper
  model_cv_results.csv        -- per-fold raw data
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

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not found. Run: pip install xgboost")


def calc_adj_r2(r2, n, p):
    denom = n - p - 1
    if denom <= 0:
        return float("nan")
    return 1.0 - (1.0 - r2) * (n - 1) / denom


def run_enhanced_evaluation():
    SEP  = "=" * 70
    DASH = "-" * 70

    print(SEP)
    print("  ENHANCED MODEL EVALUATION PIPELINE")
    print(SEP)

    dataset_path = "advanced_fused_dataset.csv"
    print("\n[1] Loading: " + dataset_path)
    df = pd.read_csv(dataset_path)

    TARGET   = "alzheimer_risk_score"
    FEATURES = [c for c in df.columns if c != TARGET]
    X = df[FEATURES].values
    y = df[TARGET].values
    n_samples, n_features = X.shape

    print("    Samples  : " + str(n_samples))
    print("    Features : " + str(n_features))
    print("    Target   : " + TARGET +
          "  range [{:.4f}, {:.4f}]".format(y.min(), y.max()))

    # ---- Models ---------------------------------------------------------------
    models = {
        "Linear Regression" : LinearRegression(),
        "Random Forest"     : RandomForestRegressor(
                                  n_estimators=100, max_depth=7,
                                  random_state=42, n_jobs=-1),
        "Decision Tree"     : DecisionTreeRegressor(random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, n_jobs=-1)

    # ---- 5-Fold CV ------------------------------------------------------------
    N_FOLDS = 5
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    print("\n[2] Running {}-Fold CV on {} models...".format(N_FOLDS, len(models)))
    print()

    all_results   = {}
    all_fold_data = []

    for name, model in models.items():
        t0 = time.time()
        print("    " + name + " ...", end="", flush=True)

        r2_list, adj_list, mae_list, rmse_list, mse_list = [], [], [], [], []

        for fi, (tr, te) in enumerate(kf.split(X)):
            model.fit(X[tr], y[tr])
            yp  = model.predict(X[te])
            n   = len(y[te])
            r2  = float(r2_score(y[te], yp))
            adj = calc_adj_r2(r2, n, n_features)
            mae = float(mean_absolute_error(y[te], yp))
            mse = float(mean_squared_error(y[te], yp))
            rm  = float(np.sqrt(mse))

            r2_list.append(r2);  adj_list.append(adj)
            mae_list.append(mae); rmse_list.append(rm); mse_list.append(mse)

            all_fold_data.append({
                "Model": name, "Fold": fi + 1,
                "R2": round(r2, 6), "Adj_R2": round(adj, 6),
                "MAE": round(mae, 6), "RMSE": round(rm, 6), "MSE": round(mse, 6),
            })

        elapsed = time.time() - t0
        print("  done ({:.1f}s)".format(elapsed))

        all_results[name] = {
            "R2_mean"    : float(np.mean(r2_list)),
            "R2_std"     : float(np.std(r2_list)),
            "AdjR2_mean" : float(np.nanmean(adj_list)),
            "AdjR2_std"  : float(np.nanstd(adj_list)),
            "MAE_mean"   : float(np.mean(mae_list)),
            "MAE_std"    : float(np.std(mae_list)),
            "RMSE_mean"  : float(np.mean(rmse_list)),
            "RMSE_std"   : float(np.std(rmse_list)),
            "MSE_mean"   : float(np.mean(mse_list)),
            "MSE_std"    : float(np.std(mse_list)),
            "Time_s"     : round(elapsed, 2),
        }

    # ---- Summary Table --------------------------------------------------------
    rows = []
    for mn, m in all_results.items():
        rows.append({
            "Model"   : mn,
            "R2_mean" : "{:.4f}".format(m["R2_mean"]),
            "R2_std"  : "+/-{:.4f}".format(m["R2_std"]),
            "AdjR2"   : "{:.4f}".format(m["AdjR2_mean"]),
            "MAE"     : "{:.6f}".format(m["MAE_mean"]),
            "RMSE"    : "{:.6f}".format(m["RMSE_mean"]),
            "MSE"     : "{:.6f}".format(m["MSE_mean"]),
            "Time(s)" : m["Time_s"],
        })
    sdf = pd.DataFrame(rows)
    sdf["_s"] = [all_results[m]["R2_mean"] for m in all_results]
    sdf = sdf.sort_values("_s", ascending=False).drop(columns="_s").reset_index(drop=True)
    sdf.insert(0, "Rank", range(1, len(sdf) + 1))

    # ---- Interpretation lines -------------------------------------------------
    ranked = list(sdf["Model"])
    best   = all_results[ranked[0]]

    interp = [
        "Best Model   : " + ranked[0],
        "  R2         : {:.4f} +/- {:.4f}  ({:.1f}% variance explained)".format(
            best["R2_mean"], best["R2_std"], best["R2_mean"] * 100),
        "  Adjusted R2: {:.4f}  (corrected for {} features)".format(
            best["AdjR2_mean"], n_features),
        "  MAE        : {:.6f}  +/- {:.6f}".format(best["MAE_mean"],  best["MAE_std"]),
        "  RMSE       : {:.6f}  +/- {:.6f}".format(best["RMSE_mean"], best["RMSE_std"]),
        "  MSE        : {:.6f}  +/- {:.6f}".format(best["MSE_mean"],  best["MSE_std"]),
        "",
    ]
    for mn in ranked[1:]:
        m = all_results[mn]
        interp.append("{:<26}: R2={:.4f} +/-{:.4f}  Delta={:.4f}".format(
            mn, m["R2_mean"], m["R2_std"], best["R2_mean"] - m["R2_mean"]))

    # ---- Console Output -------------------------------------------------------
    print()
    print(SEP)
    print("  RESULTS -- 5-FOLD CROSS-VALIDATION")
    print(SEP)
    print(sdf.to_string(index=False))
    print()
    print(DASH)
    print("  INTERPRETATION")
    print(DASH)
    for line in interp:
        print("  " + line)
    print()

    # ---- Save CSV -------------------------------------------------------------
    pd.DataFrame(all_fold_data).to_csv("model_cv_results.csv", index=False)
    print("[3] Per-fold data saved -> model_cv_results.csv")

    # ---- Build Text Report ----------------------------------------------------
    rpt = [
        SEP,
        "  ENHANCED MODEL EVALUATION REPORT",
        "  Dataset : {}  Samples={} Features={}".format(dataset_path, n_samples, n_features),
        "  CV      : {}-Fold (shuffle=True, seed=42)".format(N_FOLDS),
        "  Models  : {}".format(", ".join(models.keys())),
        SEP, "",
        "SECTION I -- COMPARISON TABLE", DASH,
        sdf.to_string(index=False),
        "", DASH, "SECTION II -- DETAILED METRICS", DASH,
    ]

    for mn, m in all_results.items():
        rpt += [
            "", "  " + mn,
            "    R2          : {:.6f}  +/- {:.6f}".format(m["R2_mean"],    m["R2_std"]),
            "    Adjusted R2 : {:.6f}  +/- {:.6f}".format(m["AdjR2_mean"], m["AdjR2_std"]),
            "    MAE         : {:.6f}  +/- {:.6f}".format(m["MAE_mean"],   m["MAE_std"]),
            "    RMSE        : {:.6f}  +/- {:.6f}".format(m["RMSE_mean"],  m["RMSE_std"]),
            "    MSE         : {:.6f}  +/- {:.6f}".format(m["MSE_mean"],   m["MSE_std"]),
            "    Train Time  : {}s".format(m["Time_s"]),
        ]

    rpt += [
        "", DASH, "SECTION III -- INTERPRETATION", DASH,
    ] + ["  " + l for l in interp]

    rpt += [
        "", DASH, "SECTION IV -- IEEE PAPER-READY TABLE", DASH, "",
        "| Rank | Model               | R2 (mean+/-std)     | AdjR2  | MAE       | RMSE      | MSE       |",
        "|------|---------------------|---------------------|--------|-----------|-----------|-----------|",
    ]
    for _, row in sdf.iterrows():
        m = all_results[row["Model"]]
        rpt.append("| {:<4} | {:<19} | {:<19} | {:<6} | {:<9} | {:<9} | {:<9} |".format(
            int(row["Rank"]), row["Model"],
            "{:.4f} +/- {:.4f}".format(m["R2_mean"], m["R2_std"]),
            "{:.4f}".format(m["AdjR2_mean"]),
            "{:.6f}".format(m["MAE_mean"]),
            "{:.6f}".format(m["RMSE_mean"]),
            "{:.6f}".format(m["MSE_mean"]),
        ))

    rpt += [
        "",
        "  All values = mean +/- std across 5 folds.",
        "  Higher R2/AdjR2 is better. Lower MAE/RMSE/MSE is better.",
        "", SEP, "  END OF REPORT", SEP,
    ]

    with open("enhanced_model_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(rpt))

    print("[4] Report saved -> enhanced_model_results.txt")
    print()
    print(SEP)
    print("  DONE -- Ready for research paper.")
    print(SEP)


if __name__ == "__main__":
    run_enhanced_evaluation()
