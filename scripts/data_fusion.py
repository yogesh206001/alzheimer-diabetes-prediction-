import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def load_and_reduce_darwin(darwin_path):
    print(f"Loading Alzheimer DARWIN dataset from {darwin_path}...")
    darwin_df = pd.read_csv(darwin_path)
    
    # We want to extract the following 6 means across the 25 tasks for each row:
    # total_time, pressure_mean, pressure_var, mean_speed_on_paper, gmrt_on_paper, mean_jerk_on_paper
    # And 1 global variability feature: std of total_time across tasks.
    
    # Collect column names for each task (1 to 25)
    total_time_cols = [f'total_time{i}' for i in range(1, 26)]
    pressure_mean_cols = [f'pressure_mean{i}' for i in range(1, 26)]
    pressure_var_cols = [f'pressure_var{i}' for i in range(1, 26)]
    speed_on_paper_cols = [f'mean_speed_on_paper{i}' for i in range(1, 26)]
    gmrt_on_paper_cols = [f'gmrt_on_paper{i}' for i in range(1, 26)]
    jerk_on_paper_cols = [f'mean_jerk_on_paper{i}' for i in range(1, 26)]
    
    reduced_df = pd.DataFrame()
    
    # 6 Meaningful Aggregated Mean Features
    reduced_df['mean_total_time'] = darwin_df[total_time_cols].mean(axis=1)
    reduced_df['mean_pressure_mean'] = darwin_df[pressure_mean_cols].mean(axis=1)
    reduced_df['mean_pressure_var'] = darwin_df[pressure_var_cols].mean(axis=1)
    reduced_df['mean_speed_on_paper'] = darwin_df[speed_on_paper_cols].mean(axis=1)
    reduced_df['mean_gmrt_on_paper'] = darwin_df[gmrt_on_paper_cols].mean(axis=1)
    reduced_df['mean_jerk_on_paper'] = darwin_df[jerk_on_paper_cols].mean(axis=1)
    
    # 1 Global Variability Feature
    reduced_df['std_total_time'] = darwin_df[total_time_cols].std(axis=1)
    
    # Optionally keep the class label for reference if needed (Patient P or Healthy H)
    reduced_df['alzheimer_label'] = darwin_df['class'].apply(lambda x: 1 if x.strip() == 'P' else 0)
    
    return reduced_df

def load_diabetes():
    print("Fetching Pima Indians Diabetes Dataset...")
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    diabetes_df = pd.read_csv(url, header=None, names=columns)
    
    # Simple cleaning: replace 0s with NaN for specific clinical columns, then impute with median
    cols_with_invalid_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    diabetes_df[cols_with_invalid_zeros] = diabetes_df[cols_with_invalid_zeros].replace(0, np.nan)
    diabetes_df.fillna(diabetes_df.median(), inplace=True)
    
    return diabetes_df

def run_fusion():
    darwin_path = "DARWIN.csv"
    if not os.path.exists(darwin_path):
        raise FileNotFoundError(f"{darwin_path} not found in the current directory.")
        
    alzheimer_df = load_and_reduce_darwin(darwin_path)
    diabetes_df = load_diabetes()
    
    # Normalize both datasets feature-wise using MinMaxScaler [0, 1]
    print("Normalizing features...")
    scaler_alz = MinMaxScaler()
    alz_scaled = pd.DataFrame(scaler_alz.fit_transform(alzheimer_df), columns=alzheimer_df.columns)
    
    scaler_diab = MinMaxScaler()
    diab_scaled = pd.DataFrame(scaler_diab.fit_transform(diabetes_df), columns=diabetes_df.columns)
    
    # Synthetic Data Fusion: We want to align rows. Let's create a robust dataset of N=1000
    print("Performing synthetic data fusion column-wise using bootstrapping...")
    N_SAMPLES = 1000
    
    # Sample independently from both datasets to ensure we do NOT assume the same patients
    alz_synthetic = alz_scaled.sample(n=N_SAMPLES, replace=True, random_state=42).reset_index(drop=True)
    diab_synthetic = diab_scaled.sample(n=N_SAMPLES, replace=True, random_state=42 + 1).reset_index(drop=True)
    
    # Merge column-wise
    fused_df = pd.concat([diab_synthetic, alz_synthetic], axis=1)
    
    # Create the alzheimer_risk_score
    print("Engineering alzheimer_risk_score feature...")
    
    # We formulate a realistic medical relationship constraint:
    # 1. Age increases risk
    # 2. Chronic glucose and Diabetes Diagnosis increases risk
    # 3. Handwriting cognitive markers directly map to risk:
    #    - More total time (slower processing)
    #    - Higher jerkiness (loss of fine motor control)
    #    - Higher variability in time (std_total_time)
    
    # Weights for the synthetic combination
    w_age = 0.25
    w_glucose = 0.15
    w_outcome = 0.20
    w_time = 0.15
    w_jerk = 0.10
    w_std = 0.15
    
    raw_risk_score = (
        (fused_df['Age'] * w_age) +
        (fused_df['Glucose'] * w_glucose) +
        (fused_df['Outcome'] * w_outcome) +
        (fused_df['mean_total_time'] * w_time) +
        (fused_df['mean_jerk_on_paper'] * w_jerk) +
        (fused_df['std_total_time'] * w_std)
    )
    
    # In some random cases, healthy people score high/low. Let's add slight gaussian noise for realism 
    # to avoid a perfectly deterministic mechanical relationship, preserving statistical variance.
    noise = np.random.normal(scale=0.05, size=N_SAMPLES)
    raw_risk_score += noise
    
    # Final Normalize Score between 0 and 1
    fused_df['alzheimer_risk_score'] = MinMaxScaler().fit_transform(raw_risk_score.values.reshape(-1, 1))

    # Output to CSV
    output_file = "fused_dataset.csv"
    fused_df.to_csv(output_file, index=False)
    print(f"Fusion complete! Data saved to {output_file}")

    # Generate Feature Description
    desc = """Unified Synthetic Dataset: Diabetes and Alzheimer Cognitive Features

This dataset contains clinical features fused with reduced cognitive measurements, generating a holistic view for predicting medical risk. Below is the data dictionary:

# Diabetes Clinical Features (Normalized)
- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index
- DiabetesPedigreeFunction: Diabetes pedigree function (genetic indicator)
- Age: Age in years
- Outcome: Class variable (0 = No Diabetes, 1 = Diabetes)

# Cognitive DARWIN Features (Normalized)
These features represent mean values calculated across 25 distinct writing/drawing tasks.
- mean_total_time: Average total duration taken to complete handwriting tasks.
- mean_pressure_mean: Average pen pressure exerted on the drawing tablet.
- mean_pressure_var: Mean of the variance in pen pressure across tasks.
- mean_speed_on_paper: Average speed of the pen while making contact with the tablet.
- mean_gmrt_on_paper: Global Mean Reaction Time - duration pen spends moving actively on the tablet.
- mean_jerk_on_paper: Average jerkiness (rate of change of acceleration), indicating motor functioning loss.
- std_total_time: (Global Variability) The standard deviation of the patient's completion times across all 25 tasks. Captures inconsistency in cognitive/motor performance.
- alzheimer_label: The original diagnostic class from the DARWIN dataset (0 = Healthy, 1 = Patient).

# Engineered Target Score
- alzheimer_risk_score: A synthetically engineered variable scaled [0,1]. Defined mathematically by increasing Age, Glucose, and Diabetes Outcome, linearly combined with longer reaction times, more motor jerkiness, and high global inconsistency (std_total_time) in drawing metrics.
"""
    with open("feature_description.txt", "w") as f:
        f.write(desc)
    print("Feature descriptions saved to feature_description.txt")

if __name__ == "__main__":
    run_fusion()
