import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
from .features import *


def roc(model, X_test, y_test):
    """
    Plot ROC curve.
    """

    # Compute predicted probabilities: y_pred_prob
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Generate ROC curve values: fpr, tpr, thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    # Plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve (AUC = %.3f)' % roc_auc_score(y_test, y_pred_prob))
    plt.show()


def roc_cv(title: str, model, X: pd.DataFrame, y: pd.Series, n_splits=5) -> list:
    """
    Plot ROC curve with cross-validation.

    Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html

    Returns
    -------
    Mean cross validtion AUC
    auc_cv, auc_std : list
        auc_cv: The AUC mean from cross validation
        auc_std: The AUC standard deviation from cross validation
    """
    cv = StratifiedKFold(n_splits=n_splits, random_state=42)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    plt.subplots(figsize=(10, 6))

    i = 0
    for train, test in cv.split(X, y):
        probas_ = model.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.3f)' % (i, roc_auc))

        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    auc_cv = auc(mean_fpr, mean_tpr)
    auc_std = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (auc_cv, auc_std),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC: {}'.format(title))
    plt.legend(loc="lower right")
    plt.show()

    return [auc_cv, auc_std]


def numeric_vs_target(df: pd.DataFrame, target: str):
    """
    Plot histograms split by target category for every numeric feature.
    
    Parameters
    ----------
    df : pd.DataFrame
        A dataframe containing numeric features and the binary target variable.
    target: str
        The column name of the target variable in df.
    """
    numeric_features = list(df.select_dtypes(include=['number']).columns)
    if target in numeric_features:
        numeric_features.remove(target)

    df_tidy = df[numeric_features + [target]].melt(id_vars=target)
    g = sns.FacetGrid(df_tidy, col='variable', hue=target, col_wrap=4, sharex=False, sharey=False)
    g.map(sns.distplot, 'value', kde=False).add_legend()


def categorical_vs_target(df: pd.DataFrame, target: str, n_largest: int = 10):
    """
    Plot count bar charts grouped by target category for every categorical feature.
    
    Parameters
    ----------
    df : pd.DataFrame
        A dataframe containing numeric features and the binary target variable.
    target: str
        The column name of the target variable in df.
    n_largest: str
        Show only the n largest categories and bin the others into an 'other' category to reduce chart height.
    """
    categorical_features = get_categorical_features(df)
    if target in categorical_features:
        categorical_features.remove(target)

    # Encode all as categorical (creates a copy of the dataframe)
    df = df[categorical_features + [target]].astype('category')

    # Only keep n largest categories, group smaller categories into 'other'
    for feature in categorical_features:
        value_counts = df[feature].value_counts()
        largest = list(value_counts.nlargest(n_largest).index)
        if len(value_counts) > n_largest:
            df[feature].cat.add_categories('other', inplace=True)
            df.loc[~df[feature].isin(largest), feature] = 'other'

        df[feature].cat.remove_unused_categories(inplace=True)

    ncols = 2
    nrows = int(np.ceil(len(categorical_features) / 2))
    figw, figh = 16.0, 4 + (nrows-1) * 1.5 + nrows * n_largest * 0.2
    plt.subplots(nrows, ncols, figsize=(figw, figh))
    # plt.subplots_adjust(left=2/figw, right=2-1/figw, bottom=1/figh, top=2-1/figh)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    for i, feature in enumerate(categorical_features):
        plt.subplot(nrows, ncols, i + 1)
        sns.countplot(data=df, y=feature, hue=target, orient='v')
        plt.xlabel('')
        plt.ylabel('')
        plt.title(feature)
