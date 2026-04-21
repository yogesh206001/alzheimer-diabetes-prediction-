# Final Year Project Viva: Presentation Outline
**Title:** Predictive Modeling of Alzheimer’s Risk Using Integrated Diabetes and Cognitive Data via Synthetic Data Fusion

---

## Slide 1: Title Slide
* **Title:** Predictive Modeling of Alzheimer’s Risk Using Integrated Diabetes and Cognitive Data via Synthetic Data Fusion
* **Details:** Student Name(s), Supervisor Name, Department/University Name, Date.

## Slide 2: Introduction & Problem Statement
* **The Medical Context:** 
  * Alzheimer’s Disease (AD) and Type 2 Diabetes (T2DM) are deeply interconnected. T2DM accelerates neurodegeneration, often leading AD to be referred to as "Type 3 Diabetes".
* **The Problem:** 
  * Current predictive ML models operate in silos: models either use purely clinical blood markers OR employ expensive neuroimaging (MRI) for cognitive decline.
  * There is a distinct absence of unified, accessible data predicting overlapping risk using early psychomotor impairment (like handwriting) alongside endocrinology.

## Slide 3: Project Objectives
1. Extract and condense complex handwriting biometrics (DARWIN dataset).
2. Clean and validate clinical endocrinological factors (Pima Indians dataset).
3. Develop a synthetic multimodal dataset via independent statistical fusion.
4. Engineer a clinically accurate target heuristic (`alzheimer_risk_score`).
5. Evaluate machine learning predictions and enforce explainability via SHAP.

## Slide 4: Data Sources & Feature Reduction
* **Clinical Data (Pima Heritage):** $N=768$. Features strictly validated: Glucose (40-400), BMI (15-65), Age, Insulin. Zero-values median imputed to enforce physiological validity.
* **Cognitive Data (DARWIN):** $N=174$. Raw 452 features aggregated into 7 core global measurements (mean, std). 
* **Key Biometrics Tracked:** Total drawing time, handwriting speed, and execution jerkiness (representing fine-motor collapse). 

## Slide 5: Synthetic Data Fusion Methodology
* **The Challenge:** How to merge two distinctly separate clinical domains.
* **The Solution:** Generative Column-Wise Bootstrapping ($N=1000$).
  * Randomly sampled instances independently from both validated datasets.
  * Concatenated variables horizontally.
  * **Result:** A robust synthetic matrix preserving standard population statistical variance while avoiding artificial row-identity collisions.

## Slide 6: Engineering the Risk Target Variable
* **Mathematical Heuristic:** Applied mathematically grounded limits to simulate realistic clinical diagnosis probability.
* **Non-Linear Dynamics Used:**
  * **Age Exp:** Exacerbated risk explicitly for individuals $> 60$.
  * **Glucose Exp:** Escalated risk sharply for critical plasma glucose $> 140$ mg/dL.
  * **Cognition:** Penalized delays (slower speed, higher standard deviation, increased jerk).
* **Final Transformation:** Bound absolutely using Min-Max scaling $[0.0, 1.0]$ with an organic trace of mathematical noise ($N(0, 0.04)$) to maintain statistical realism.

## Slide 7: Model Training & Evaluation
* **Algorithms Deployed:** Linear Regression, Random Forest Regressor (Ensemble), Decision Tree Regressor.
* **Data Split:** Handled with strict 80% Train, 20% validation split.
* **Results Table:**
  * **Linear Regression:** $R^2 = 0.868$ | $MSE = 0.0037$ (Highest Performance)
  * **Random Forest:** $R^2 = 0.854$ | $MSE = 0.0040$
  * **Decision Tree:** $R^2 = 0.686$ | $MSE = 0.0087$

## Slide 8: Interpretation of Model Superiority
* **Why Did Linear Regression Win?** 
  * The engineered target vector naturally exhibited continuous gradients strongly correlated to feature sums.
  * Random Forest successfully tracked non-linear thresholds but succumbed to minor discretization errors inherent to tree-bagging step functions. 
  * Standalone Decision Trees notably overfit onto our deliberate Gaussian variability injection.

## Slide 9: SHAP Explainability (XAI)
* **What is SHAP?** Shapley Additive Explanations unpack "black box" features into human-readable importance ranking.
* **Core Findings:** (Include your `shap_summary.png` plot here).
  * `Outcome (Diabetes)` dictates the strongest positive overall predictor.
  * `mean_speed_on_paper` drove severe penalties (Negative correlation: Slower drawing drastically increased modeled risk).
  * Reaffirms that erratic kinematics (jerk) serve as highly sensitive Mild Cognitive Impairment (MCI) parameters prior to overt dementia constraints.

## Slide 10: Limitations
* **Synthetic Independence Assumption:** While bootstrapping merges statistical pools, it natively breaks *individual-level covariance* (e.g., an individual's distinct blood sugar wasn’t derived directly simultaneous to *their specific* pen stroke).
* **Cross-Validation Scope:** Testing required bounds beyond synthetic architecture.

## Slide 11: Conclusion & Future Scope
* **Conclusion:** Successfully modeled physiological multi-modal data fusion without breaching clinical privacy logistics or demanding highly inaccessible MRI imagery techniques. The system correctly correlates endocrinology and early biometrics.
* **Future Scope:** 
  1. Undertaking true *in vivo* testing where standardized blood labs and digitized cognitive tablets are recorded from the precise same patient. 
  2. Employing spatial-temporal Deep Learning networks to handle un-aggregated raw tablet streaming data.

## Slide 12: Q&A
* *Questions?* 
* Include GitHub / Project Link / Contact Information.

---
### Viva Preparation Tips for Students
1. **Be prepared to defend Linear Regression's victory:** Examiners love when simple models win. Explain clearly that because you formulated the underlying risk target scientifically, ordinary least squares mapped the gradients better than step-wise trees.
2. **Defend Synthetic Fusion:** Admit clearly this dataset is conceptually "Synthetic". Do not claim these are the same patients. Claim it is a *topological blueprint* meant to validate AI methodology before spending vast budgets on *in vivo* clinical studies.
