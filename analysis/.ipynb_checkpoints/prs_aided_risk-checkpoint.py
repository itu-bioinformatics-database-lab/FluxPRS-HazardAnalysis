import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def calculate_risk_score_with_best_prs(y, risk_scores, sample_ids, best_prs_path, prs_combined):
    biomarker_df = pd.DataFrame({
        'Diagnosis': y,
        'risk_score': risk_scores,
        'sample_id_biomarker': sample_ids
    })

    prs = pd.read_csv(best_prs_path)

    prs["sample_id"] = prs["sample_id"].astype(str)
    biomarkers_df["sample_id_biomarker"] = biomarkers_df["sample_id_biomarker"].astype(str)
    combined_df = pd.merge(biomarker_df, prs, left_on="sample_id_biomarker", right_on="sample_id", how="inner")

    scaler = MinMaxScaler()
    combined_df["z_score"] = scaler.fit_transform(combined_df[["score"]])

    x_combined = np.vstack([combined_df["risk_score"], combined_df["z_score"]]).T  # Features
    x_combined = StandardScaler().fit_transform(x_combined)
    y = combined_df["Diagnosis"]

    model = LogisticRegression()

    if len(np.unique(y)) > 1:
        model.fit(x_combined, y)
        combined_risk = model.predict_proba(x_combined)[:, 1]  # Combined risk scores
        combined_df["combined_score"] = combined_risk
        combined_df.to_csv(
            prs_combined,
            index=False,
            sep=",",
            columns=['sample_id_biomarker','score', 'risk_score', 'Diagnosis', 'z_score']
        )
        combined_df.to_csv(prs_combined, index=False, sep=",", columns = ['sample_id_biomarker', 'score', 'risk_score', 'Diagnosis', 'z_score'])
    else:
        combined_df["combined_score"] = np.nan
    
    #model.fit(x_combined, y)
    #combined_risk = model.predict_proba(x_combined)[:, 1]  # Combined risk scores
    #combined_df["combined_score"] = combined_risk
    

    return combined_df

def calculate_risk_score_with_prs(y, risk_scores, sample_ids, prs_df, prs_combined):
    biomarker_df = pd.DataFrame({
        'Diagnosis': y,
        'risk_score': risk_scores,
        'sample_id_biomarker': sample_ids
    })

    prs = prs_df
    prs["sample_id"] = prs["sample_id"].astype(str)
    biomarker_df["sample_id_biomarker"] = biomarker_df["sample_id_biomarker"].astype(str)
    combined_df = pd.merge(biomarker_df, prs, left_on="sample_id_biomarker", right_on="sample_id", how="inner")

    scaler = MinMaxScaler()
    combined_df["z_score"] = scaler.fit_transform(combined_df[["score"]])

    x_combined = np.vstack([combined_df["risk_score"], combined_df["z_score"]]).T  # Features
    x_combined = StandardScaler().fit_transform(x_combined)
    y = combined_df["Diagnosis"]

    model = LogisticRegression()
    if len(np.unique(y)) > 1:
        model.fit(x_combined, y)
        combined_risk = model.predict_proba(x_combined)[:, 1]  # Combined risk scores
        combined_df["combined_score"] = combined_risk
        combined_df.to_csv(
            prs_combined,
            index=False,
            sep=",",
            columns=['score', 'risk_score', 'Diagnosis', 'z_score']
        )
    else:
        combined_df["combined_score"] = np.nan
        combined_df.to_csv(prs_combined, index=False, sep=",", columns = ['score', 'risk_score', 'Diagnosis', 'z_score'])
    #model.fit(x_combined, y)
    #combined_risk = model.predict_proba(x_combined)[:, 1]  # Combined risk scores
    #combined_df["combined_score"] = combined_risk
    


    return combined_df
