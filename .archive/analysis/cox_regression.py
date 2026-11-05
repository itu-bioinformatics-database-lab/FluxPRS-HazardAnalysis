from lifelines import CoxPHFitter
import pandas as pd
import numpy as np


# file: survival_analysis/cox_model.py

import pandas as pd
from lifelines import CoxPHFitter
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

def cox_regression_with_matching(df, duration_df, sample_ids_df,
                                 variance_thresh=1e-5, corr_thresh=0.95):
    """
    Perform Cox regression with matched sample IDs and feature selection

    Parameters:
    df (pd.DataFrame): Contains features, 'Diagnosis' (event indicator: 0/1), and 'order_index'
    duration_df (pd.DataFrame): Columns: 'Sample ID' and 'duration'
    sample_ids_df (pd.DataFrame): Contains 'Sample ID's matching df's 'order_index' sequence
    variance_thresh (float): Minimum variance required to keep a feature
    corr_thresh (float): Maximum correlation allowed between features

    Returns:
    pd.Series: Risk scores from Cox model
    pd.DataFrame: Filtered survival data used for fitting
    """
    df = df.copy()
    df['Sample ID'] = sample_ids_df.values.flatten()

    df["Sample ID"] = df["Sample ID"].astype(str)
    duration_df["Sample ID"] = duration_df["Sample ID"].astype(str)
    merged = pd.merge(df, duration_df, on='Sample ID', how='inner')
    if len(merged) < len(df):
        print(f"Warning: {len(df) - len(merged)} samples dropped during merging")

    # Drop NA
    merged = merged.dropna()

    # Feature columns
    features = [col for col in merged.columns if col not in ['Diagnosis', 'order_index', 'Sample ID', 'duration']]
    X = merged[features]

    # Step 1: Drop low variance features
    selector = VarianceThreshold(threshold=variance_thresh)
    X_var = selector.fit_transform(X)
    kept_features = X.columns[selector.get_support(indices=True)]
    X = pd.DataFrame(X_var, columns=kept_features, index=merged.index)

    # Step 2: Drop highly correlated features
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = [column for column in upper.columns if any(upper[column] > corr_thresh)]
    X = X.drop(columns=drop_cols)

    # Step 3: Standardize
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Step 4: Final data
    survival_data = pd.concat([merged[['duration', 'Diagnosis']], X_scaled], axis=1)

    # Fit Cox model
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(survival_data, duration_col='duration', event_col='Diagnosis', show_progress=False)

    risk_scores = cph.predict_partial_hazard(survival_data)

    return risk_scores, survival_data


def compute_hazard_ratios(risk_scores, survival_data):
    # Map Diagnosis to binary: 1 for "AD", 0 for "Control"
    survival_data = survival_data.copy()
    survival_data['Diagnosis'] = survival_data['Diagnosis'].apply(lambda x: 0 if x == "Control" else 1)

    # Add risk scores to data
    survival_data['risk_score'] = risk_scores

    top_n = int(len(survival_data) * 0.10)

    top_indices = survival_data['risk_score'].nlargest(top_n).index

    survival_data['high_risk'] = 0

    survival_data.loc[top_indices, 'high_risk'] = 1

    # Refit Cox model with high_risk as the only predictor
    df_hr = survival_data[['duration', 'Diagnosis', 'high_risk']].copy()

    cph_hr = CoxPHFitter()
    cph_hr.fit(df_hr, duration_col='duration', event_col='Diagnosis')

    summary = cph_hr.summary

    coef = summary.loc['high_risk', 'coef']
    exp_coef = summary.loc['high_risk', 'exp(coef)']
    p_value = summary.loc['high_risk', 'p']

    print(f"Log(HR) (coef): {coef}")
    print(f"Hazard Ratio (exp(coef)): {exp_coef}")
    print(f"P-value: {p_value}")

    return exp_coef

