import pandas as pd


def exclude_columns_all_zero(df):
    df = df.loc[:, (df != 0).any(axis=0)]
    return df

def remove_unnecessary_columns(df):
    """
    Remove columns at indices 2, 3, and 4 from the DataFrame.
    Race, PMI, Braak

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with specified columns removed
    """
    columns_to_drop = df.columns[[2, 3, 4]]
    return df.drop(columns=columns_to_drop)
