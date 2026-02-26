import pandas as pd
import numpy as np
from .utils import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_metabolite_levels_for_all_ad_control(
        metabolite_levels_path
):
    df = pd.read_csv(metabolite_levels_path)
    df = exclude_columns_all_zero(df)
    df = remove_unnecessary_columns(df)

    df = df.dropna()
    df = df.drop("Gender", axis='columns') # can be removed
    df['Diagnosis'] = (df['Diagnosis'] == 'AD').astype(int)

    sample_ids = df['Sample ID']
    df = df.drop(columns=['Sample ID'])
    df["order_index"] = range(len(df))
    order_index = df['order_index']
    return df, sample_ids, order_index

def z_standardize(x_train, x_test):
    scaler = StandardScaler()
    numeric_cols = x_train.select_dtypes(include=[np.number]).columns

    x_train[numeric_cols] = scaler.fit_transform(x_train[numeric_cols])
    x_test[numeric_cols] = scaler.transform(x_test[numeric_cols])

    return x_train, x_test

def transform_pathway_file():
    pathway_df = pd.read_csv('data/raw/X_alzheimer_pathway.csv')
    metabolite_levels_df = pd.read_csv('data/Metabolon_HD4_intersecting_with_SNP_270_samples_just_AD_and_Normal.csv')

    first_cols = metabolite_levels_df.iloc[:, :5]
    pathway_df = pathway_df.iloc[:, 1:]
    pathway_df.con
    pathway_df.to_csv('data/interim/pathway_270_ad_normal.csv')


def prepare_pathway_df_for_all_ad_control():
    df = pd.read_csv('data/pathway_270_ad_normal.csv')
    df = exclude_columns_all_zero(df)
    df = remove_unnecessary_columns(df)

    df = df.dropna()
    df = df.drop("Gender", axis='columns') # can be removed
    df['Diagnosis'] = (df['Diagnosis'] == 'AD').astype(int)

    sample_ids = df['Sample ID']
    df = df.drop(columns=['Sample ID'])
    df["order_index"] = range(len(df))
    order_index = df['order_index']
    return df, sample_ids, order_index








