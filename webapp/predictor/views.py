from django.shortcuts import render
import os
import joblib
import numpy as np
import pandas as pd

# ── Resolve base directory (one level above webapp/) ────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Load artefacts once at startup ──────────────────────────────────────────
_MODEL  = joblib.load(os.path.join(BASE_DIR, 'final_model.pkl'))
_SCALER = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
_META   = joblib.load(os.path.join(BASE_DIR, 'feature_meta.pkl'))

FEATURE_ORDER  = _META['feature_order']
DEFAULTS       = _META['feature_defaults']
FEAT_IMP       = _META['feature_importances']   # {feature_name: importance_score}

# ── Friendly display names for top-factor explanation ───────────────────────
FRIENDLY = {
    'Outcome'                  : 'Clinical Diabetes Diagnosis',
    'Age'                      : 'Advanced Age',
    'Glucose'                  : 'Elevated Blood Glucose',
    'BMI'                      : 'Body Mass Index (BMI)',
    'Insulin'                  : 'Insulin Levels',
    'DiabetesPedigreeFunction'  : 'Genetic Diabetes Risk',
    'BloodPressure'            : 'Blood Pressure',
    'Pregnancies'              : 'Number of Pregnancies',
    'SkinThickness'            : 'Skin Thickness',
    'mean_total_time'          : 'Slower Psychomotor Execution',
    'std_total_time'           : 'High Task Time Variability',
    'mean_pressure_mean'       : 'Pen Pressure Pattern',
    'mean_pressure_var'        : 'Pressure Irregularity',
    'mean_speed_on_paper'      : 'Reduced Handwriting Speed',
    'mean_gmrt_on_paper'       : 'GMRT Motor Rhythm',
    'mean_jerk_on_paper'       : 'Erratic Motor Movement (Jerk)',
}


def _safe_float(request, key, default):
    try:
        return float(request.POST.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def index(request):
    return render(request, 'predictor/index.html')


def predict(request):
    if request.method != 'POST':
        return render(request, 'predictor/index.html')

    # ── 1. Parse inputs ──────────────────────────────────────────────────────
    raw = {
        'Pregnancies'             : _safe_float(request, 'pregnancies',        DEFAULTS['Pregnancies']),
        'Glucose'                 : _safe_float(request, 'glucose',            DEFAULTS['Glucose']),
        'BloodPressure'           : _safe_float(request, 'blood_pressure',     DEFAULTS['BloodPressure']),
        'SkinThickness'           : _safe_float(request, 'skin_thickness',     DEFAULTS['SkinThickness']),
        'Insulin'                 : _safe_float(request, 'insulin',            DEFAULTS['Insulin']),
        'BMI'                     : _safe_float(request, 'bmi',               DEFAULTS['BMI']),
        'DiabetesPedigreeFunction': _safe_float(request, 'diabetes_pedigree',  DEFAULTS['DiabetesPedigreeFunction']),
        'Age'                     : _safe_float(request, 'age',               DEFAULTS['Age']),
        'Outcome'                 : _safe_float(request, 'outcome',           0.0),
        'mean_total_time'         : _safe_float(request, 'mean_total_time',    DEFAULTS['mean_total_time']),
        'mean_pressure_mean'      : _safe_float(request, 'mean_pressure_mean', DEFAULTS['mean_pressure_mean']),
        'mean_pressure_var'       : _safe_float(request, 'mean_pressure_var',  DEFAULTS['mean_pressure_var']),
        'mean_speed_on_paper'     : _safe_float(request, 'mean_speed_on_paper',DEFAULTS['mean_speed_on_paper']),
        'mean_gmrt_on_paper'      : _safe_float(request, 'mean_gmrt_on_paper', DEFAULTS['mean_gmrt_on_paper']),
        'mean_jerk_on_paper'      : _safe_float(request, 'mean_jerk_on_paper', DEFAULTS['mean_jerk_on_paper']),
        'std_total_time'          : _safe_float(request, 'std_total_time',     DEFAULTS['std_total_time']),
    }

    # ── 2. Scale using the trained MinMaxScaler (raw → [0,1]) ────────────────
    input_df     = pd.DataFrame([raw], columns=FEATURE_ORDER)
    scaled_input = _SCALER.transform(input_df)

    # ── 3. Predict & clip ────────────────────────────────────────────────────
    raw_pred       = _MODEL.predict(scaled_input)[0]
    risk_score_val = float(np.clip(raw_pred, 0.0, 1.0))
    percentage     = round(risk_score_val * 100, 1)

    # ── 4. Risk category ─────────────────────────────────────────────────────
    if risk_score_val < 0.30:
        category   = "Low Risk"
        badge_css  = "badge-low"
        bar_css    = "bar-low"
        text_css   = "score-low"
    elif risk_score_val < 0.60:
        category   = "Moderate Risk"
        badge_css  = "badge-medium"
        bar_css    = "bar-medium"
        text_css   = "score-medium"
    else:
        category   = "High Risk"
        badge_css  = "badge-high"
        bar_css    = "bar-high"
        text_css   = "score-high"

    # ── 5. Top contributing factors (XGBoost feature_importances_) ───────────
    # Multiply global importance by the scaled input value to get per-sample contribution
    scaled_vals = scaled_input[0]
    contributions = {
        feat: float(FEAT_IMP.get(feat, 0)) * float(scaled_vals[i])
        for i, feat in enumerate(FEATURE_ORDER)
    }
    top_factors = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:4]

    explain_lines = []
    for feat, score in top_factors:
        friendly = FRIENDLY.get(feat, feat)
        if score > 0.05:
            level = "strongly"
        elif score > 0.01:
            level = "moderately"
        else:
            level = "mildly"
        explain_lines.append({
            'label' : friendly,
            'text'  : f"{friendly} {level} elevated your risk profile.",
            'score' : round(score * 100, 1),
        })

    # ── 6. Clinical recommendations ──────────────────────────────────────────
    recommendations = []
    glucose = raw['Glucose']
    age     = raw['Age']
    outcome = raw['Outcome']
    mtt     = raw['mean_total_time']

    if outcome == 1.0:
        recommendations.append("Diabetes diagnosis detected. Strict glycaemic management is critical to reduce neuro-inflammation linked to Alzheimer's pathology.")
    if glucose > 140:
        recommendations.append("Severely elevated glucose (>140 mg/dL). Consult an endocrinologist for hyperglycaemic control.")
    elif glucose > 100:
        recommendations.append("Pre-diabetic glucose range detected. Dietary modification and regular monitoring are advised.")
    if age >= 65:
        recommendations.append("Age 65+ is a primary risk factor. Regular cognitive screening via MoCA or MMSE is recommended annually.")
    if mtt > DEFAULTS['mean_total_time'] * 1.2:
        recommendations.append("Elevated psychomotor task execution time detected — may indicate early cognitive slowing. Neurological consultation advised.")
    if risk_score_val >= 0.60:
        recommendations.append("High composite risk score. Comprehensive neurological evaluation with a specialist is strongly recommended.")

    if not recommendations:
        recommendations.append("No significant risk flags detected. Maintain a healthy lifestyle with regular cognitive engagement, balanced diet, and physical activity.")

    context = {
        'percentage'      : percentage,
        'category'        : category,
        'badge_css'       : badge_css,
        'bar_css'         : bar_css,
        'text_css'        : text_css,
        'recommendations' : recommendations,
        'explain_lines'   : explain_lines,
        'model_used'      : 'XGBoost Regressor',
        'cv_r2'           : '0.8705 ± 0.0084',
    }
    return render(request, 'predictor/result.html', context)
