import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def facet_numeric(df: pd.DataFrame, numeric_features: list, target: str):
    """
    Plot histograms split by target category for every numeric feature.
    """
    df_tidy = df[numeric_features + [target]].melt(id_vars=target)
    g = sns.FacetGrid(df_tidy, col='variable', hue=target, col_wrap=4, sharex=False, sharey=False)
    g = g.map(sns.distplot, 'value', kde=False).add_legend()
    
    
def facet_categorical(df: pd.DataFrame, categorical_features: list, target: str, n_largest=10):
    """
    Plot count bar charts grouped by target category for every categorical feature.
    """
    df = df[categorical_features + [target]].copy()

    # Only keep n largest categories, group smaller categories into 'other'
    for feature in categorical_features:
        value_counts = df[feature].value_counts()
        largest = list(value_counts.nlargest(n_largest).index)
        df.loc[~df[feature].isin(largest), feature] = 'other'
    
    ncols=2
    nrows=int(np.ceil(len(categorical_features) / 2))
    figw, figh = 16.0, 4 + nrows*n_largest*0.3
    plt.subplots(nrows, ncols, figsize=(figw, figh))
    #plt.subplots_adjust(left=2/figw, right=2-1/figw, bottom=1/figh, top=2-1/figh)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    for i, feature in enumerate(categorical_features):
        plt.subplot(nrows, ncols, i + 1)
        sns.countplot(data=df, y=feature, hue=target, orient='v')
        plt.xlabel('')
        plt.ylabel('')
        plt.title(feature)