import pandas as pd


def get_numeric_features(df: pd.DataFrame) -> list:
    return list(df.select_dtypes(include=["number"]).columns)


def get_categorical_features(df: pd.DataFrame) -> list:
    return list(df.select_dtypes(include=["object", "category"]).columns)
