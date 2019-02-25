import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

__all__ = ['plot_residuals_vs_fitted_values',
           'plot_residuals_vs_predicted_values',
           'pp_plot', 'qq_plot',
           'variance_inflation_factors']


def plot_residuals_vs_fitted_values(residual_values, fitted_values,
                                    ax=None):
    """Plot a model's residual values (y-axis) on the fitted values
    (x-axis).

    The plot is a useful diagnostic to assess whether there is
    heteroskedasticity or outliers in the data.

    Args:
        residual_values (array): array of residuals from a model
        fitted_values (array): array of fitted values (y-fit)
        ax (Axes object): Matplotlib Axes object (optional)

    Returns:
        Figure object
    """

    if ax is None:
        ax = plt.gca()
    fig = sns.regplot(residual_values, fitted_values,
                      ax=ax, fit_reg=False)
    ax.grid(True, linewidth=0.5)
    ax.set_title("Residuals vs Fitted Values Plot")
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    return fig


def plot_residuals_vs_predicted_values(residual_values, predicted_values,
                                       ax=None):
    """Plot a model's residual values (y-axis) on the predicted values
    (x-axis).

    The plot is a useful diagnostic to assess whether the assumption of
    linearity holds for a model.

    Args:
        residual_values (array): array of residuals from a model
        predicted_values (array): array of fitted values (y-pred)
        ax (Axes object): Matplotlib Axes object (optional)

    Returns:
        Figure object
    """
    if ax is None:
        ax = plt.gca()
    fig = sns.regplot(residual_values, predicted_values,
                      ax=ax, fit_reg=False)
    ax.grid(True, linewidth=0.5)
    ax.set_title("Residuals vs Predicted Values Plot")
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    return fig


def pp_plot(residual_values, ax=None):
    """P-P plot compares the empirical cumulative distribution function
    against the theoretical cumulative distribution function given a
    specified model.

    The plot is a useful diagnostic to assess whether the assumption of
    linearity holds for a model (more sensitive to non-linearity in the
    middle of the distribution).

    Args:
        residual_values (array): array of residuals from a model
        ax (Axes object): Matplotlib Axes object (optional)

    Returns:
        Figure object
    """

    if ax is None:
        ax = plt.gca()
    prob_plot = sm.ProbPlot(residual_values)
    fig = prob_plot.ppplot(ax=ax, color='tab:blue', markersize=4,
                           line='45')  # Figure returned is passed to subplot
    ax.grid(True, linewidth=0.5)
    ax.set_title("P-P Plot of Residuals")
    return fig


def qq_plot(residual_values, ax=None):
    """Q-Q plot compares the quantiles of a distribution against the
    quantiles of the theoretical distribution given a specified model.

    The plot is a useful diagnostic to assess whether the assumption of
    linearity holds for a model (more sensitive to non-linearity in the tails).

    Args:
        residual_values (array): array of residuals from a model
        ax (Axes object): Matplotlib Axes object (optional)

    Returns:
        Figure object
    """

    if ax is None:
        ax = plt.gca()
    prob_plot = sm.ProbPlot(residual_values)
    fig = prob_plot.qqplot(ax=ax, line='s', color='tab:blue',
                           markersize=4)  # Figure to be passed to subplot
    ax.grid(True, linewidth=0.5)
    ax.set_title("Q-Q Plot of Residuals")
    return fig


def variance_inflation_factors(X, vif_threshold=10):
    """Returns a DataFrame with variance inflation factor (VIF) values
    calculated given a matrix of regressor values (X).

    VIF values are used as indicators for multicollinearity.
    Econometrics literature typically uses a threshold of 10 to indicate
    problems with multicollinearity in a model.

    Args:
        X (pd.DataFrame): matrix of values for the regressors (one
            row per regressor).
        vif_threshold (int, optional): Defaults to 10.  The threshold
            set for assessment of multicollinearity.

    Returns:
        pd.DataFrame: columns for VIF values, their tolerance and the
            result of the heuristic.
    """
    X = sm.add_constant(X)
    # Set up variance inflation factor values:
    vif = pd.Series([1 / (1.0 - (sm.OLS(X[col].values,
                                        X.drop(columns=[col]).values))
                          .fit().rsquared)
                     for col in X],
                    index=X.columns,
                    name='VIF')
    vif = vif.drop('const')  # constant not needed for output
    # Calculate tolerance:
    tol = pd.Series(1 / vif, index=vif.index, name='1/VIF')
    vif_thres = pd.Series(vif > vif_threshold,
                          name="VIF>" + str(vif_threshold))
    # Final dataframe:
    return pd.concat([vif, tol, vif_thres], axis='columns')
