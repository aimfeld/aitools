import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from .features import *


def classify(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, test_size: int = 0.3) -> float:
    """
    Run a binary classifier given a list of numeric and categorical features.
    Plots a kfold cross-validation ROC curve and prints a classification report.
    
    Parameters
    ----------
    pipeline : Pipeline
        A classification pipeline, see create_pipeline().
    X : pd.DataFrame
        A dataframe containing feature columns.
    y : pd.Series of int, bool, str, or categorical
        A series containing the binary classification labels.
    test_size : int, default = 0.3
        The size of the holdout test dataset for evaluating the model.
    
    Returns
    -------
    auc_holdout : float
        The AUC score for the holdout test
    """
    # Make sure it works with numerical, categorical, and string labels
    y = y.astype('category').cat.codes

    # Create training and test sets
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=test_size, random_state=42)

    # Fit the pipeline to the train set
    pipeline.fit(X_train, y_train)

    auc_holdout = pipeline.score(X_holdout, y_holdout)
    print("Holdout AUC score: %.3f" % auc_holdout)

    # Predict the labels of the test set
    y_pred = pipeline.predict(X_holdout)

    # Compute metrics
    print('\nHoldout classification report:\n', classification_report(y_holdout, y_pred, digits=3))

    return auc_holdout


def create_pipeline(
    classifier,
    X: pd.DataFrame,
    numeric_prepro: Pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('encoder', MinMaxScaler())]
    ),
    categorical_prepro: Pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))]
    )) -> Pipeline:
    """
    Creates a classifier pipeline with separate preprocessing for categorical and numeric features.

    Parameters
    ----------
    classifier :
        A scikit-learn compatible classifier, e.g. RandomForestClassifier().
    X : pd.DataFrame,
        A dataframe containing feature columns.
    numeric_prepro : Pipeline, default = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('encoder', MinMaxScaler())]
        )
    categorical_prepro : Pipeline, default = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))]
        )

    Returns
    -------
    pipeline : Pipeline
        A pipeline containing the preprocessing and classification steps.
    """
    transformers = []

    numeric_features = get_numeric_features(X)
    transformers.append(('numeric_prepro', numeric_prepro, numeric_features))

    categorical_features = get_categorical_features(X)
    transformers.append(('categorical_prepro', categorical_prepro, categorical_features))

    steps = [('preprocessor', ColumnTransformer(transformers=transformers)), ('classifier', classifier)]
    pipeline = Pipeline(steps)

    return pipeline
