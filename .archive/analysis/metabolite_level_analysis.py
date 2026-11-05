import pandas as pd

from preprocessing import *
from sklearn.linear_model import LogisticRegression
from .evaluate_model import evaluate_model
from .plotter import plot_calibration_curve
from lifelines import CoxPHFitter

def analyse_with_logistic_regression(
        preprocessor: Preprocessor
):
    preprocessor.transform_custom_file()
    metabolite_levels, sample_ids, order_index = preprocessor.prepare_df()


    x = metabolite_levels.drop(columns=['Diagnosis'])
    y = metabolite_levels['Diagnosis']

    x_train, x_test, y_train, y_test, sample_id_train, sample_id_test, order_train, order_test = train_test_split(
        x, y, sample_ids, order_index,  test_size=0.3, random_state=42, stratify=y
    )

    x_train, x_test = z_standardize(x_train, x_test)

    model = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    model.fit(x_train, y_train)

    # evaluate_model(model, x_test, y_test, preprocessor.name, x_train, y_train)
    # plot_calibration_curve(model, x_test, y_test, 7, preprocessor.name)

    train_risk_scores = model.predict_proba(x_train)[:, 1]
    test_risk_scores = model.predict_proba(x_test)[:, 1]

    train_df = pd.concat([
        pd.Series(sample_id_train, name="Sample ID", index=x_train.index),
        pd.Series(order_train, name="order_index", index=x_train.index),
        pd.Series(train_risk_scores, name="Risk Score", index=x_train.index)
    ], axis=1)

    test_df = pd.concat([
        pd.Series(sample_id_test, name="Sample ID", index=x_test.index),
        pd.Series(order_test, name="order_index", index=x_test.index),
        pd.Series(test_risk_scores, name="Risk Score", index=x_test.index)
    ], axis=1)

    combined_df = pd.concat([train_df, test_df]).sort_values(by="order_index").drop(columns=["order_index"])
    y_combined = pd.concat([y_train, y_test]).loc[combined_df.index]  # Labels
    combined_risk_scores = combined_df["Risk Score"]

    return y_combined, combined_risk_scores, combined_df["Sample ID"]

def calculate_odds_ratio(risk_scores, y):
    x = risk_scores.to_frame()
    model = LogisticRegression()
    model.fit(x, y)

    coef = model.coef_[0][0]

    odds_ratio_ = np.exp(coef)
    return odds_ratio_

def create_ages_df(biomarkers_path, clinical_ages_path):

    clinical_df = pd.read_csv(clinical_ages_path, header=0)

    biomarkers_df = pd.read_csv(biomarkers_path, header=0)

    clinical_df["Sample ID"] = clinical_df["Sample ID"].astype(str)
    biomarkers_df["Sample ID"] = biomarkers_df["Sample ID"].astype(str)
    merged_df = biomarkers_df.merge(clinical_df, how="left", left_on="Sample ID", right_on="Sample ID")

    if ("Diagnosis_x" in merged_df.columns):
        merged_df["Diagnosis"] = merged_df["Diagnosis_x"]
    keep_columns = [column for column in ["Sample ID", "Gender", "Diagnosis", "age_first_ad_dx", "age_at_visit_max", "age_death", "AgeDeath"] if column in merged_df.columns]

    return merged_df[keep_columns]

def create_ages_df_with_df(df, clinical_ages_path, metabolite_levels_file_path):
    df = pd.DataFrame(df)
    df.columns = ['Sample ID']
    clinical_df = pd.read_csv(clinical_ages_path, header=0)
    metabolite_levels = pd.read_csv(metabolite_levels_file_path, header=0)
    clinical_df["Sample ID"] = clinical_df["Sample ID"].astype(str)
    df["Sample ID"] = df["Sample ID"].astype(str)
    merged_df_ = df.merge(clinical_df, how="left", left_on="Sample ID", right_on="Sample ID")

    if ("Diagnosis_x" in merged_df_.columns):
        merged_df_["Diagnosis"] = merged_df_["Diagnosis_x"]

    metabolite_levels["Sample ID"] = metabolite_levels["Sample ID"].astype(str)

    merged_df = metabolite_levels.merge(merged_df_, how="inner", left_on="Sample ID", right_on="Sample ID")

    if ("Diagnosis_x" in merged_df.columns):
        merged_df["Diagnosis"] = merged_df["Diagnosis_x"]
        
    keep_columns = [column for column in ["Sample ID", "Gender", "Diagnosis", "age_first_ad_dx", "age_at_visit_max", "age_death", "AgeDeath"] if column in merged_df.columns]
    return merged_df[keep_columns]

def prepare_duration(ages_df):

    if "age_first_ad_dx" in ages_df.columns and "age_at_visit_max" in ages_df.columns:
        age_at_first_ad_dxs = ages_df["age_first_ad_dx"]
        age_at_visit_maxs = ages_df["age_at_visit_max"]
    
        age_at_first_ad_dxs = pd.to_numeric(age_at_first_ad_dxs.replace("90+", "90"), errors='coerce')
        age_at_visit_maxs = pd.to_numeric(age_at_visit_maxs.replace("90+", "90"), errors='coerce')
    
        duration = age_at_first_ad_dxs.fillna(age_at_visit_maxs)
        duration_df = pd.DataFrame({
            "duration": duration,
            "Sample ID": ages_df["Sample ID"],
            "Diagnosis":  ages_df["Diagnosis"]
        })

    else:
        duration = pd.to_numeric(ages_df["AgeDeath"].replace("90+", "90"), errors='coerce')
        duration_df = pd.DataFrame({
            "duration": duration,
            "Sample ID": ages_df["Sample ID"],
            "Diagnosis":  ages_df["Diagnosis"]
        })
    return duration_df

def calculate_hazard_ratio(risk_scores, y, duration):
    data = pd.DataFrame({
        'duration': duration,
        'risk_score': risk_scores,
        'event': y
    })
    cph = CoxPHFitter()
    cph.fit(data, duration_col='duration', event_col='event')
    cph.print_summary()  # Shows hazard ratios, confidence intervals, p-values




