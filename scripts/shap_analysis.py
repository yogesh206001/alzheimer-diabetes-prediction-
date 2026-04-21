import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Disable headless error for plotting
import matplotlib
matplotlib.use('Agg')

def run_shap():
    print("Loading data...")
    df = pd.read_csv("advanced_fused_dataset.csv")
    
    X = df.drop('alzheimer_risk_score', axis=1)
    y = df['alzheimer_risk_score']
    
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X, y)
    
    print("Computing SHAP values...")
    # Using LinearExplainer since we established this is a linear model
    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    
    print("Generating SHAP Summary Plot...")
    # Generate summary plot
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=300, bbox_inches='tight')
    print("Saved SHAP summary plot to shap_summary.png")
    
    # Calculate Mean Absolute SHAP values for ranking
    shap_abs = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        'Feature': X.columns,
        'Mean_Abs_SHAP': shap_abs
    }).sort_values(by='Mean_Abs_SHAP', ascending=False)
    
    print("\n--- SHAP FEATURE IMPORTANCE RANKING ---")
    print(shap_df.to_string(index=False))

if __name__ == "__main__":
    run_shap()
