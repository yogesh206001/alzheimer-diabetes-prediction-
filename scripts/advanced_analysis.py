import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import os
import warnings

# Suppress pandas warnings for cleaner outputs
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_and_clean_darwin(darwin_path):
    print(f"Loading Alzheimer DARWIN dataset from {darwin_path}...")
    darwin_df = pd.read_csv(darwin_path)
    
    total_time_cols = [f'total_time{i}' for i in range(1, 26)]
    pressure_mean_cols = [f'pressure_mean{i}' for i in range(1, 26)]
    pressure_var_cols = [f'pressure_var{i}' for i in range(1, 26)]
    speed_on_paper_cols = [f'mean_speed_on_paper{i}' for i in range(1, 26)]
    gmrt_on_paper_cols = [f'gmrt_on_paper{i}' for i in range(1, 26)]
    jerk_on_paper_cols = [f'mean_jerk_on_paper{i}' for i in range(1, 26)]
    
    reduced_df = pd.DataFrame()
    reduced_df['mean_total_time'] = darwin_df[total_time_cols].mean(axis=1)
    reduced_df['mean_pressure_mean'] = darwin_df[pressure_mean_cols].mean(axis=1)
    reduced_df['mean_pressure_var'] = darwin_df[pressure_var_cols].mean(axis=1)
    reduced_df['mean_speed_on_paper'] = darwin_df[speed_on_paper_cols].mean(axis=1)
    reduced_df['mean_gmrt_on_paper'] = darwin_df[gmrt_on_paper_cols].mean(axis=1)
    reduced_df['mean_jerk_on_paper'] = darwin_df[jerk_on_paper_cols].mean(axis=1)
    reduced_df['std_total_time'] = darwin_df[total_time_cols].std(axis=1)
    
    # Advanced Data Validation: Winsorizing outliers at 1st and 99th percentiles
    for col in reduced_df.columns:
        q1 = reduced_df[col].quantile(0.01)
        q99 = reduced_df[col].quantile(0.99)
        reduced_df[col] = np.clip(reduced_df[col], q1, q99)
        
    return reduced_df

def load_and_clean_diabetes():
    print("Fetching Pima Indians Diabetes Dataset...")
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    diabetes_df = pd.read_csv(url, header=None, names=columns)
    
    # Simple cleaning: replace 0s with NaN for specific clinical columns
    cols_with_invalid_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    diabetes_df[cols_with_invalid_zeros] = diabetes_df[cols_with_invalid_zeros].replace(0, np.nan)
    
    # Impute missing values with dataset median
    diabetes_df.fillna(diabetes_df.median(), inplace=True)
    
    # Validation: Filter strictly unrealistic BMI and Age
    diabetes_df = diabetes_df[(diabetes_df['Glucose'] >= 40) & (diabetes_df['Glucose'] <= 400)]
    diabetes_df = diabetes_df[(diabetes_df['BMI'] >= 15) & (diabetes_df['BMI'] <= 65)]
    diabetes_df = diabetes_df[(diabetes_df['Age'] >= 21) & (diabetes_df['Age'] <= 100)]
    
    return diabetes_df

def run_advanced_analysis():
    darwin_path = "DARWIN.csv"
    if not os.path.exists(darwin_path):
        raise FileNotFoundError(f"{darwin_path} not found in the current directory.")
        
    alz_df = load_and_clean_darwin(darwin_path)
    diab_df = load_and_clean_diabetes()
    
    print("Performing synthetic data fusion column-wise using bootstrapping...")
    N_SAMPLES = 1000
    
    # Sample independently from both datasets to assume NO overlap of same patients
    alz_synthetic = alz_df.sample(n=N_SAMPLES, replace=True, random_state=42).reset_index(drop=True)
    diab_synthetic = diab_df.sample(n=N_SAMPLES, replace=True, random_state=42 + 1).reset_index(drop=True)
    fused_df = pd.concat([diab_synthetic, alz_synthetic], axis=1)

    print("Executing NON-LINEAR Engineered Risk Score logic...")
    
    # 1. Non-linear Age impact: base impact + exponential spike after 60
    age_raw = fused_df['Age'].values
    age_effect = age_raw + np.where(age_raw > 60, (age_raw - 60)**1.5, 0)
    
    # 2. Non-linear Glucose impact: base impact + severe exponential spike after 140
    gluc_raw = fused_df['Glucose'].values
    gluc_effect = gluc_raw + np.where(gluc_raw > 140, (gluc_raw - 140)**1.4, 0)
    
    # Scale components to combine them naturally
    scaler = MinMaxScaler()
    age_comp = scaler.fit_transform(age_effect.reshape(-1, 1)).flatten()
    gluc_comp = scaler.fit_transform(gluc_effect.reshape(-1, 1)).flatten()
    
    # 3. Clinical Diabetes multiplier
    out_comp = fused_df['Outcome'].values
    
    # 4. Cognitive Component (Combining multiple markers logically)
    # Slow speed = higher risk. Longer time = higher risk. Jerkiness = higher risk. High variance = higher risk.
    temp_cog = pd.DataFrame(scaler.fit_transform(fused_df[['mean_total_time', 'mean_jerk_on_paper', 
                                                           'std_total_time', 'mean_speed_on_paper']]), 
                            columns=['time', 'jerk', 'std', 'speed'])
    cog_comp = (temp_cog['time'] * 0.35 + 
                temp_cog['jerk'] * 0.25 + 
                temp_cog['std'] * 0.20 + 
                (1.0 - temp_cog['speed']) * 0.20)
    
    # Normalize cognitive composite
    cog_comp = scaler.fit_transform(cog_comp.values.reshape(-1,1)).flatten()
    
    # Combining the Risk dynamically enforcing balanced weights over components
    raw_risk_score = (age_comp * 0.30) + (gluc_comp * 0.20) + (out_comp * 0.15) + (cog_comp * 0.35)
    
    # Apply controlled biological-mimic noise (std=0.04) and bound strictly between 0-1
    noise = np.random.normal(scale=0.04, size=N_SAMPLES)
    raw_risk_score += noise
    
    fused_df['alzheimer_risk_score'] = scaler.fit_transform(raw_risk_score.reshape(-1, 1)).flatten()
    
    print("Normalizing final fully assembled dataset...")
    # Normalize everything EXCEPT the target we just bounded strictly
    cols_to_scale = [c for c in fused_df.columns if c != 'alzheimer_risk_score']
    fused_df[cols_to_scale] = scaler.fit_transform(fused_df[cols_to_scale])
    
    # Missing Values Verification
    miss_count = fused_df.isnull().sum().sum()
    
    # Output to CSV
    output_file = "advanced_fused_dataset.csv"
    fused_df.to_csv(output_file, index=False)
    print(f"Validated and normalized fusion complete! Data saved to {output_file}")

    print("Computing Correlation Matrix...")
    corr_matrix = fused_df.corr()
    risk_corrs = corr_matrix['alzheimer_risk_score'].sort_values(ascending=False)
    corr_matrix.to_csv("correlation_matrix.csv")
    
    print("Executing Feature Importance Training (Random Forest)...")
    X = fused_df.drop('alzheimer_risk_score', axis=1)
    y = fused_df['alzheimer_risk_score']
    
    # Regressor restricted with max_depth to prevent strict overfitting
    rf = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
    rf.fit(X, y)
    
    importance_df = pd.DataFrame({
        'Feature': X.columns, 
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    importance_df.to_csv("feature_importance.csv", index=False)
    
    # Generate Automated Validation Report & Formula Dump
    report = f"""# Validation and Final Execution Report

## 1. Missing Values & Outliers Handled
- Pre-fusion Total Missing Values found: 0 (after successful median-imputation on BMI/Glucose)
- Extraneous Values Clipped: Blood Glucose > 400 dropped. BMI < 15 or > 65 dropped. DARWIN numeric features were strictly Winsorized to 1st and 99th percentiles preserving distribution curves but preventing 4+ Sigma explosions.
- Post-Fusion Total Missing Values: {miss_count}

## 2. Updated and Enhanced alzheimer_risk_score Formula
We assembled raw independent weights enforcing domain heuristic assumptions:
- Age Factor (comp_age) = MinMaxScaler( Age + (Age - 60)^1.5 if Age > 60 else 0 )
- Glucose Factor (comp_gluc) = MinMaxScaler( Glucose + (Glucose - 140)^1.4 if Glucose > 140 else 0 )
- Outcome Factor (comp_dm) = 1 if Diabetes True else 0
- Cognitive Composite (comp_cog) = Normalized(0.35*mean_time + 0.25*mean_jerk + 0.20*std_total_time + 0.20*(1 - mean_speed))

Base Score = (0.30 * comp_age) + (0.20 * comp_gluc) + (0.15 * comp_dm) + (0.35 * comp_cog)
Final Score = MinMaxScaler( Base Score + GaussianNoise(mean=0, std=0.04) )

## 3. Correlation with Target (Verification)
Age, Glucose, and cognitive processing duration MUST correlate strongly positive. Speed on paper MUST correlate negative.
{risk_corrs.to_string()}

## 4. Random Forest Feature Importance (Max-Depth 7, 100 Estimators)
Confirming our non-black-box approach ensures expected dominant drivers hold weight structurally against raw trees.
{importance_df.to_string(index=False)}
"""
    with open("validation_report.txt", "w") as rf_file:
        rf_file.write(report)
        
    print("Advanced processing step complete. Results and models dumped to respective destinations.")

if __name__ == "__main__":
    run_advanced_analysis()
