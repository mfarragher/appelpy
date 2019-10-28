import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
import seaborn as sns
import matplotlib.pyplot as plt

__all__ = ['BadApples',
           'plot_residuals_vs_fitted_values',
           'plot_residuals_vs_predicted_values',
           'pp_plot', 'qq_plot',
           'variance_inflation_factors',
           'heteroskedasticity_test',
           'partial_regression_plot']


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
    chart_plot = sns.regplot(residual_values, fitted_values,
                             ax=ax, fit_reg=False)
    fig = chart_plot.figure
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
    chart_plot = sns.regplot(residual_values, predicted_values,
                             ax=ax, fit_reg=False)
    fig = chart_plot.figure
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


def partial_regression_plot(appelpy_model_object, df, regressor,
                            annotate_results=False, ax=None):
    """Also known as the added variable plot, the partial regression plot
    shows the effect of adding another regressor (independent variable)
    to a regression model.

    Args:
        appelpy_model_object: the object that contains the info about a model
            fitted with Appelpy.  e.g. for OLS regression the object would
            be of the type appelpy.linear_model.OLS.
        df (pd.DataFrame): dataframe used as an input in the original model.
        regressor (str): the 'added variable', present in df, to examine
            in the partial regression plot.
        annotate_results (Bool): Defaults to False.  Annotate the plot with
            the coefficient (b) and t-value of the regressor from the 'full'
            model that now includes the regressor. (optional)
        ax (Axes object): Matplotlib Axes object (optional)

    Returns:
        Figure object
    """

    X_list = appelpy_model_object.X.columns.tolist()
    y_list = [appelpy_model_object.y.name]

    if ax is None:
        ax = plt.gca()

    if (regressor not in X_list and regressor in df.columns
            and not df[regressor].isnull().any()):
        model_ylist = (sm.OLS(df[y_list], sm.add_constant(df[X_list]))
                       .fit(disp=0))
        model_var = (sm.OLS(df[regressor], sm.add_constant(df[X_list]))
                     .fit(disp=0))
        chart_plot = sns.regplot(model_var.resid, model_ylist.resid,
                                 ci=None, truncate=True,
                                 line_kws={'color': 'red'})
        if annotate_results:
            model_full = (sm.OLS(model_ylist.resid,
                                 sm.add_constant(df[X_list + [regressor]]))
                          .fit(disp=0))
            ax.set_title(('Partial regression plot: {}\n(b={:.4f}, t={:.3f})'
                          .format(regressor,
                                  model_full.params.loc[regressor],
                                  model_full.tvalues.loc[regressor])))
        else:
            ax.set_title('Partial regression plot: {}'.format(regressor))
        fig = chart_plot.figure
        ax.grid()
        ax.set_ylabel('e({} | X)'.format(y_list[0]))
        ax.set_xlabel('e({} | X)'.format(regressor))
        return fig
    elif (regressor not in X_list and regressor in df.columns
          and df[regressor].isnull().any()):
        raise ValueError("""Null values found in the column for the regressor.
        Account for them before calling a partial regression plot.""")
    elif regressor in X_list:
        cols = list(set(X_list) - set([regressor]))
        model_ylist = (sm.OLS(df[y_list], sm.add_constant(df[cols]))
                       .fit(disp=0))
        model_var = (sm.OLS(df[regressor], sm.add_constant(df[cols]))
                     .fit(disp=0))
        chart_plot = sns.regplot(model_var.resid, model_ylist.resid,
                                 ci=None, truncate=True,
                                 line_kws={'color': 'red'})
        if annotate_results:
            ax.set_title(('Partial regression plot: {}\n(b={:.4f}, t={:.3f})'
                          .format(regressor,
                                  appelpy_model_object.params.loc[regressor],
                                  appelpy_model_object.tvalues.loc[regressor])))
        else:
            ax.set_title('Partial regression plot: {}'.format(regressor))
        fig = chart_plot.figure
        ax.grid()
        ax.set_ylabel('e({} | X)'.format(y_list[0]))
        ax.set_xlabel('e({} | X)'.format(regressor))
        return fig
    else:
        raise ValueError("Regressor not found in the dataset.")


class BadApples:
    """The BadApples class takes an appelpy model object and can provide
    diagnostics of extreme observations.  It calculates measures of
    influence, leverage and outliers in the model.

    With k independent variables and n observations, the class calculates
    these measures and flags observations as extreme based on the specified
    heuristics:

    INFLUENCE:
    - dfbeta (for each independent variable): DFBETA diagnostic.
        Extreme if val > 2 / sqrt(n)
    - dffits (for each independent variable): DFFITS diagnostic.
        Extreme if val > 2 * sqrt(k / n)
    - cooks_d: Cook's distance.  Extreme if val > 4 / n
    LEVERAGE:
    - leverage: value from hat matrix diagonal.  Extreme if
        val > (2*k + 2) / n
    OUTLIERS:
    - resid_standardized: standardized residual.  Extreme if
        |val| > 2, i.e. approx. 5% of observations will be
        flagged.
    - resid_studentized: studentized residual.  Extreme if
        |val| > 2, i.e. approx. 5% of observations will be
        flagged.

    These measures are available in Statsmodels via the get_influence
    method on a regression object, but Appelpy does a decomposition of
    these measures into leverage, outliers and influence.

    Args:
        appelpy_model_object: the object that contains the info about a model
            fitted with Appelpy.  e.g. for OLS regression the object would
            be of the type appelpy.linear_model.OLS.

    Methods:
        fit: calculate the influence measures & heuristics and store the
            information in the object.
        show_extreme_observations: return a dataframe that shows values that
            have high influence, high leverage or are outliers, based on
            at least one heuristic.
        plot_leverage_vs_residuals_squared: plot the model's leverage values
            on the normalized residuals squared.  The equivalent command in
            Stata is lvr2plot.

    Attributes:
        measures_influence (pd.DataFrame): dataframe with measures of
            influence, including: dfbeta for each independent variable;
            dffits for each independent variable; Cook's distance (cooks_d)
        measures_leverage (pd.Series): a series with the leverage values by
            observation.  These values are from the diagonal of the hat
            matrix.
        measures_outliers (pd.DataFrame): dataframe with residual measures,
            namely the standardized residuals (resid_standardized) and
            Studentized residuals (resid_studentized).
        indices_high_influence (list): list of indices of X (i.e.
            observations) that have high influence on at least one
            measure.
        indices_high_leverage (list): list of indices of X (i.e.
            observations) that have high leverage based on a heuristic.
        indices_outliers (list): list of indices of X (i.e.
            observations) that are viewed as outliers based on a heuristic.

    Attributes (auxiliary - used to store arguments or inputs):
        appelpy_model_object
        y
        X

    """

    def __init__(self, appelpy_model_object):
        if not appelpy_model_object.is_fitted:
            raise ValueError("Ensure model is fitted first.")

        # Inputs and model info
        self._appelpy_model_object = appelpy_model_object
        self._y = appelpy_model_object.y
        self._X = appelpy_model_object.X

    @property
    def y(self):
        """pd.Series: endogenous / dependent variable"""
        return self._y

    @property
    def X(self):
        """pd.DataFrame: exogenous / independent variables"""
        return self._X

    @property
    def measures_influence(self):
        """pd.DataFrame: measures of influence per observation in the
        model."""
        return self._measures_influence

    @property
    def measures_leverage(self):
        """pd.Series: leverage values per observation in the model."""
        return self._measures_leverage

    @property
    def measures_outliers(self):
        """pd.DataFrame: measures of outliers per observation in the model."""
        return self._measures_outliers

    @property
    def indices_high_influence(self):
        """list: indices that have a high value (based on a heuristic) for
        at least one of the influence measures."""
        return self._indices_high_influence

    @property
    def indices_high_leverage(self):
        """list: indices that have a high value (based on a heuristic) for
        the leverage measure."""
        return self._indices_high_leverage

    @property
    def indices_outliers(self):
        """list: indices that have a high value (based on a heuristic) for
        at least one of the outlier measures."""
        return self._indices_outliers

    def _calculate(self):
        # Statsmodels object
        influence_obj = self._appelpy_model_object.results.get_influence()

        # Get Statsmodels calcs - but tidy them up for Appelpy
        influence = influence_obj.summary_frame()
        influence.columns = ['_'.join(['dfbeta', col[4:]])
                             if col.startswith('dfb_')
                             else col
                             for col in influence]

        # Decomposition of Statsmodels influence object
        self._measures_outliers = influence[['standard_resid',
                                             'student_resid']].copy()
        self._measures_outliers.columns = ['resid_standardized',
                                           'resid_studentized']
        self._measures_leverage = influence['hat_diag'].rename(
            'leverage').copy()
        self._measures_influence = influence.drop(
            columns=['standard_resid', 'student_resid', 'hat_diag']).copy()

        pass

    def _calculate_heuristics(self):
        if self._X.ndim == 1:
            k = 1
            n = len(self._X)
        else:
            n, k = self._X.shape
        # Leverage points:
        self._indices_high_leverage = (self._measures_leverage[(self._measures_leverage >
                                                                (2*k + 2) /
                                                                n)]
                                       .index.tolist())

        # Outlier points:
        outlier_points = []
        for col in self._measures_outliers.columns:
            indices = (self._measures_outliers[self._measures_outliers[col].abs() > 2]
                       .index.tolist())
            outlier_points.extend(indices)
        # Return list of indices (without duplicates)
        outlier_points = np.unique(outlier_points).tolist()
        self._indices_outliers = outlier_points

        # Influence points:
        influence_points = []
        for col in self._measures_influence.columns[(self._measures_influence.columns
                                                     .str.startswith('dfbeta'))].tolist():
            indices = (self._measures_influence[(self._measures_influence[col].abs() >
                                                 2 / np.sqrt(n))]
                       .index.tolist())
            influence_points.extend(indices)
        for col in self._measures_influence.columns[(self._measures_influence.columns
                                                     .str.startswith('dffits'))].tolist():
            indices = (self._measures_influence[(self._measures_influence[col].abs() >
                                                 2 * np.sqrt(k / n))]
                       .index.tolist())
            influence_points.extend(indices)
        indices_cooks = (self._measures_influence[(self._measures_influence['cooks_d'] >
                                                   4 / n)]
                         .index.tolist())
        influence_points.extend(indices_cooks)
        # Return list of indices (without duplicates)
        influence_points = np.unique(influence_points).tolist()
        self._indices_high_influence = influence_points

        pass

    def fit(self, printing=False):
        """Calculate the influence, outlier and leverage measures for the
        given model.  The method call also calculates the heuristics for
        determining which observations have 'extreme' values for the measures.

        Args:
            printing (bool, optional): display print statements to show
                progress of function calls. Defaults to False.

        Returns:
            Instance of the BadApples object, with the influence, outlier and
            leverage measures stored as attributes.
        """
        if printing:
            print('Calculating influence measures...')
        self._calculate()
        self._calculate_heuristics()
        if printing:
            print('Calculations saved to object.')

        return self

    def show_extreme_observations(self):
        """Return a dataframe that shows values that have high influence,
        high leverage or are outliers, based on at least one heuristic.

        Returns:
            pd.DataFrame: a subset of extreme observations from X.
        """
        extreme_indices_list = list(set().union(self._indices_high_leverage,
                                                self._indices_outliers,
                                                self._indices_high_influence))

        df = pd.concat([self._y, self._X], axis='columns')
        return df[df.index.isin(extreme_indices_list)].copy()

    def _calculate_leverage_vs_residuals_squared(self, rescale=False):
        df = pd.DataFrame(index=self._measures_leverage.index,
                          columns=['leverage', 'resid_score'])
        df['leverage'] = self._measures_leverage

        if rescale:
            df['resid_score'] = pd.Series((self._measures_outliers['resid_standardized'] ** 2 /
                                           len(self._measures_outliers)),
                                          index=self._measures_outliers.index)
        else:
            df['resid_score'] = pd.Series(self._measures_outliers['resid_standardized'] ** 2,
                                          index=self._measures_outliers.index)

        return df

    def plot_leverage_vs_residuals_squared(self, annotate=False,
                                           rescale=False, ax=None):
        """Produce a scatterplot of observations' leverage values
        (y-axis) on their normalized residuals squared (x-axis).  In
        Stata, the equivalent plot would be produced via the lvr2plot
        command.

        The horizontal line signifies the mean leverage value
        and the vertical line signifies the mean normalized residual
        squared.

        Scatterplot annotations are not shown by default.
        Set rescale=True to divide the normalized residual squared
        values by the number of observations.

        Args:
            annotate (bool, optional): Annotate the scatterplot points
                with the index value if a point has a residual higher
                than average or leverage higher than average.  Defaults
                to False.
            rescale (bool, optional): Divide the normalized residual squares
                by the number of observations.  Stata does that rescaling
                in its lvr2plot.  Defaults to False.
            ax (Axes object, optional): Matplotlib Axes object

        Returns:
            Figure object
        """
        if ax is None:
            ax = plt.gca()

        leverage_mean = self._measures_leverage.mean()

        plot_df = self._calculate_leverage_vs_residuals_squared(
            rescale=rescale)
        if rescale:
            resid_mean = plot_df['resid_score'].mean()
            path_c = plt.scatter(plot_df['resid_score'],
                                 self._measures_leverage.values)
            ax.set_xlabel(
                r"Normalized $resid^{2}$ / total observations")
        else:
            resid_mean = plot_df['resid_score'].mean()
            path_c = plt.scatter(plot_df['resid_score'],
                                 self._measures_leverage.values)
            ax.set_xlabel(r"Normalized $resid^{2}$")
        ax.set_ylabel("Leverage")
        ax.axvline(resid_mean, linestyle='--', color='gray')
        ax.axhline(leverage_mean, linestyle='--', color='gray')
        ax.set_title("Leverage vs Normalized Residuals Squared Plot")

        # Annotate only the points with high leverage or high residual:
        if annotate:
            plot_df['leverage_hi'] = np.where(
                plot_df['leverage'] > leverage_mean, 1, 0)
            plot_df['resid_score_hi'] = np.where(
                plot_df['resid_score'] > resid_mean, 1, 0)
            plot_df['annotate'] = np.where(
                (plot_df['leverage_hi'] | plot_df['resid_score_hi']), 1, 0)

            for index, row in plot_df[plot_df['annotate'] == 1].iterrows():
                ax.annotate(index, (row['resid_score'], row['leverage']),
                            xytext=(3, 3),
                            textcoords='offset points')

        fig = path_c.figure

        return fig


def heteroskedasticity_test(test_name, appelpy_model_object,
                            regressors_subset=None):
    """Return the results of a heteroskedasticity test given a model.

    Supported tests:
    - 'breusch_pagan': equivalent to Stata's `hettest` command.
    - 'breusch_pagan_studentized': equivalent to default behaviour of the
        bptest command in R.
    - 'white': equivalent to Stata's `imtest, white` command.

    Args:
        test_name (str): either 'breusch_pagan', 'breusch_pagan_studentized'
            or 'white'.
        appelpy_model_object: the object that contains the info about a model
            fitted with Appelpy.  e.g. for OLS regression the object would
            be of the type appelpy.linear_model.OLS.
        regressors_subset (list, optional): For breusch_pagan, this can be
            set so that the test runs on a subset of regressors. Defaults to
            None.

    Raises:
        ValueError: Choose one of 'breusch_pagan', 'breusch_pagan_studentized'
            or 'white' as a test name.
        ValueError: Check the regressors_subset items were used in the model.

    Returns:
        test_statistic, p_value: the test statistic and the corresponding
            p-value.
    """
    if test_name not in ['breusch_pagan', 'breusch_pagan_studentized',
                         'white']:
        raise ValueError(
            """Choose one of 'breusch_pagan', 'breusch_pagan_studentized' or
            'white' as a test name.""")

    if test_name == 'breusch_pagan':
        # Get residuals (from model object or run again on a regressors subset)
        if regressors_subset:
            if not set(regressors_subset).issubset(set(appelpy_model_object.X.columns)):
                raise ValueError(
                    'Regressor(s) not recognised in dataset.  Check the list given to the function.')
            reduced_model = sm.OLS(appelpy_model_object.y,
                                   sm.add_constant(appelpy_model_object.X[regressors_subset]))
            reduced_model_results = reduced_model.fit()
            sq_resid = (reduced_model_results.resid ** 2).to_numpy()
        else:
            sq_resid = (appelpy_model_object.resid ** 2).to_numpy()

        # Scale the residuals
        scaled_sq_resid = sq_resid / sq_resid.mean()
        y_hat = appelpy_model_object.results.fittedvalues.to_numpy()

        # Model of norm resid on y_hat
        aux_model = sm.OLS(scaled_sq_resid, sm.add_constant(y_hat)).fit()

        # Calculate test stat and pval
        lm = aux_model.ess / 2
        pval = sp.stats.chi2.sf(lm, 1)  # dof=1
        return lm, pval

    if test_name == 'breusch_pagan_studentized':
        if regressors_subset:
            if not set(regressors_subset).issubset(set(appelpy_model_object.X.columns)):
                raise ValueError(
                    'Regressor(s) not recognised in dataset.  Check the list given to the function.')
            reduced_model = sm.OLS(appelpy_model_object.y,
                                   sm.add_constant(appelpy_model_object.X[regressors_subset]))
            reduced_model_results = reduced_model.fit()
            lm, pval, _, _ = sms.het_breuschpagan(reduced_model_results.resid,
                                                  reduced_model_results.model.exog)
            return lm, pval
        else:
            lm, pval, _, _ = sms.het_breuschpagan(appelpy_model_object.results.resid,
                                                  appelpy_model_object.results.model.exog)
            return lm, pval

    if test_name == 'white':
        if regressors_subset:
            print("Ignoring regressors_subset.  White test will use original regressors.")
        white_test = sms.het_white(appelpy_model_object.resid,
                                   sm.add_constant(appelpy_model_object.X))
        return white_test[0], white_test[1]  # lm, pval
