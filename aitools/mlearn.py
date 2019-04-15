import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from .plot import roc_cv
from .features import *


def classify(classifier_label: str, classifier,
             X: pd.DataFrame, y: pd.Series,
             num_imputer=SimpleImputer(strategy='median'),
             cat_imputer=SimpleImputer(strategy='constant', fill_value='missing'),
             num_scaler=MinMaxScaler(),
             cat_encoder=OneHotEncoder(handle_unknown='ignore', sparse=False),
             plot_roc_cv: bool = True, test_size: int = 0.3) -> list:
    """
    Run a binary classifier given a list of numeric and categorical features.
    Plots a kfold cross-validation ROC curve and prints a classification report.
    
    Parameters
    ----------
    classifier_label : str
        A label for the classifier.
    classifier :
        A scikit learn classifier, e.g. RandomForestClassifier().
    X : pd.DataFrame
        A dataframe containing feature columns.
    y : pd.Series of int, bool, or str
        A series containing the binary classification labels.
    num_imputer : default = SimpleImputer(strategy='median')
        Imputer for numeric features.
    cat_imputer : default = SimpleImputer(strategy='constant', fill_value='missing')
        Imputer for categorical features.
    num_scaler : default = MinMaxScaler
        Scaler which is applied to numeric features.
    cat_encoder : default = OneHotEncoder(handle_unknown='ignore', sparse=False)
        Label encoder for categorical featuers.
    plot_roc_cv : bool, default = True
        Plot cross-validated ROC.
    test_size : int, default = 0.3
        The size of the holdout test dataset for evaluating the model.
    
    Returns
    -------
    model, AUC score : list
    
    Example
    -------
    model, auc_score = mlearn.num_cat_classify('Random forest', RandomForestClassifier(), X, y, categorical_features, numeric_features)
    """
    numeric_features = get_numeric_features(X)
    categorical_features = get_categorical_features(X)

    # We create the preprocessing pipelines for both numeric and categorical data.
    numeric_transformer = Pipeline(steps=[
        ('imputer', num_imputer),
        ('scaler', num_scaler)])

    categorical_transformer = Pipeline(steps=[
        ('imputer', cat_imputer),
        ('encoder', cat_encoder)])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            (classifier_label, classifier)])

    # If y-labels are given as strings, recode them into int
    if y.dtype == 'O':
        y = y.astype('category').cat.codes

    # Plot ROC cross validation curves
    if plot_roc_cv:
        roc_cv(classifier_label, model, X, y)

    # Create training and test sets
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=test_size, random_state=42)

    # Fit the pipeline to the train set
    model.fit(X_train, y_train)

    AUC_holdout_score = model.score(X_holdout, y_holdout)
    print("%s: Holdout AUC score: %.3f" % (classifier_label, AUC_holdout_score))

    # Predict the labels of the test set
    y_pred = model.predict(X_holdout)

    # Compute metrics
    print('\n%s: Holdout classification report:\n' % classifier_label,
          classification_report(y_holdout, y_pred, digits=3))

    return [model, AUC_holdout_score]
