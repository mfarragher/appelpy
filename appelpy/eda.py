"""Methods for general exploratory data analysis.

These are for analysis on datasets, rather than analysis on models.
"""

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns


def statistical_moments(df, kurtosis_fisher=True):
    """Produce a dataframe with the four main statistical moments calculated
    for each continuous variable specified in a given dataframe.

    The dataframe has the moments stored in columns and each continuous
    variable stored in the index.

    The columns for each probability distribution:
    - 'mean': the first moment
    - 'var': the second central moment
    - 'skew': the third standardized moment
    - 'kurtosis': Defaults to 'fisher'.  The fourth standardized moment.
        Fisher's kurtosis (kurtosis excess) is used by default, otherwise
        Pearson's kurtosis is used.

    Args:
        df (pd.DataFrame): dataframe with numerical columns

    Returns:
        pd.DataFrame: shape (# numerical regressors, 4)
    """
    df_numeric = df.select_dtypes(include=np.number)
    df_stats = pd.DataFrame(columns=['mean', 'var', 'skew', 'kurtosis'],
                            index=df_numeric.columns)
    for col in df_numeric.columns:
        df_stats.loc[col, 'mean'] = np.mean(df_numeric[col].dropna())
        df_stats.loc[col, 'var'] = np.var(df_numeric[col].dropna())
        df_stats.loc[col, 'skew'] = sp.stats.skew(df_numeric[col].dropna())
        if kurtosis_fisher:
            df_stats.loc[col, 'kurtosis'] = sp.stats.kurtosis(
                df_numeric[col].dropna(), fisher=True)
        else:
            df_stats.loc[col, 'kurtosis'] = sp.stats.kurtosis(
                df_numeric[col].dropna(), fisher=False)
    return df_stats


def correlation_heatmap(df, font_size=12, ax=None):
    """Produce annotated heatmap for lower triangle of correlation matrix,
    given a specified dataframe.

    Args:
        df (pd.DataFrame): dataframe with numerical columns
        font_size (int, optional): Defaults to 12.  The font size of the
            correlation values displayed in the heatmap cells
        ax (Axes object): Matplotlib Axes object (optional)

    Returns:
        Figure object
    """
    if ax is None:
        plt.gca()

    # Correlation matrix via Pandas (numeric data only)
    corr_matrix = df.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Store heatmap from mask
    fig = sns.heatmap(corr_matrix, mask=mask,
                      cmap='RdBu_r', cbar_kws={"shrink": .6},
                      annot=True, annot_kws={"size": font_size},
                      vmax=1, vmin=-1, linewidths=.5,
                      square=True, ax=ax)
    return fig
