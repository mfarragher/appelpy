import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW
import matplotlib.pyplot as plt
from .diagnostics import (plot_residuals_vs_fitted_values,
                          plot_residuals_vs_predictor_values,
                          pp_plot, qq_plot)
from .utils import _df_input_conditions
__all__ = ['WLS', 'OLS']


class WLS:
    """Weighted Least Squares (WLS) model.

    Regression pipeline:
        - Initialise model object (specify source dataframe and X & y
            columns)
        - Fit model:
            - Run model and store estimates as model object attributes
            - Run model on standardized X and y and store estimates in
                attributes.

    Args:
        df (pd.DataFrame): Pandas DataFrame that contains the data to use
            for modelling.  Each row is an observation and each column is
            either an independent variable (regressor / exogenous
            variable) or a dependent variable (endogenous variable).
        y_list (list): list containing the dependent variable,
            e.g. ['points']
        X_list (list): list containing one or more regressors,
            e.g. ['exper', 'age', 'coll', 'expersq']
        w (array-like object): Weights for each observation
        cov_type (str, optional): Defaults to 'nonrobust'.  Standard errors
            type.
            - 'nonrobust': standard errors not adjusted for
                heteroskedasticity.
            - 'HC1': robust standard errors (used for 'robust' argument in
                Stata)
            - 'HC2': robust standard errors obtained via HCCM estimates.
                Recommended by Long & Ervin (1999) where number of
                observations >= 250.
            - 'HC3': robust standard errors obtained via HCCM estimates.
                Recommended by by Long & Ervin (1999) where number of
                observations < 250.
        cov_options (dict, optional): Specify keyword arguments for cov_type.
            This is a wrapper around the cov_kwds parameter in Statsmodels,
            although column lists are used for dict values instead of Pandas
            objects.
            e.g. when specifying a group 'state' for clustered standard errors,
            the form is cov_options={'groups': ['state']},
            instead of cov_kwds={'groups': df['state']}.
        alpha (float, optional): Defaults to 0.05.  The significance level
            used for reporting confidence intervals in the model summary.

    Methods:
        fit: fit the model and store the results in the model object.
        diagnostic_plot: create a plot for regression diagnostics,
            e.g. P-P plot, Q-Q plot, residuals vs predicted values plot,
            residuals vs fitted values plot.
        predict: predict the value(s) of given example(s) based on the fitted
            model.
        significant_regressors: return a list of regressors that are
            significant for a given significance level (alpha).

    Attributes (main):
        model_selection_stats (dict): Model selection stats such as Root MSE,
            AIC, BIC, R-squared and R-squared (adjusted) stored in one place.
        results_output (Statsmodels Summary): Summary object that displays the
            main results from the model.  Suitable for printing in various
            formats.
        results_output_standardized (pd Styler): Summary object that
            displays the main results from the model, in the form of
            standardized estimates.  NOT suitable for print statements.
        results (Statsmodels object): stores information from the Statsmodels
            OLS regression.
        resid (pd.Series): residuals obtained from the fitted model.
        resid_standardized (pd.Series): standardized residuals obtained
            from the fitted model.
        y (pd.Series): contains the values of the independent variable
            for the observations that were used to fit the model.
        y_standardized (pd.Series): y in the form of standardized
            estimates.
        X (pd.DataFrame): contains the values of the regressors for the
            observations used to fit the model.
        X_standardized (pd.DataFrame): X in the form of
            standardized estimates.
        is_fitted (Boolean): indicator for whether the model has been fitted.

    Attributes (auxiliary - used to store arguments):
        df
        y_list
        X_list
        cov_type
        cov_options
        w
        alpha
    """

    def __init__(self, df, y_list, X_list, *, w=None,
                 cov_type='nonrobust', cov_options=None, alpha=0.05):
        """Initializes the WLS model object."""
        # Model inputs (attributes from arguments):
        self._df = df
        [y_name] = y_list  # sequence unpacking in order to make Series
        self._y = df[y_name]  # Pandas Series
        if len(X_list) == 1:
            [x_name] = X_list
            self._X = df[x_name].to_frame()  # Pandas dataframe
        else:
            self._X = df[X_list]  # Pandas dataframe
        self._y_list, self._X_list = y_list, X_list
        self._cov_type = cov_type
        self._cov_options = cov_options if cov_options else {}
        if w is None:
            self._w = pd.Series(np.ones(len(self._X)))
        else:
            self._w = w
        self._alpha = alpha
        self._is_fitted = True

    # MODEL INPUTS
    # These should be immutable
    @property
    def df(self):
        """pd.DataFrame: source dataset"""
        return self._df

    @property
    def y(self):
        """pd.Series: endogenous / dependent variable"""
        return self._y

    @property
    def X(self):
        """pd.DataFrame: exogenous / independent variables"""
        return self._X

    @property
    def w(self):
        """pd.Series: weight for each observation"""
        return self._w

    @property
    def y_list(self):
        """list: argument for the endogenous / dependent variable"""
        return self._y_list

    @property
    def X_list(self):
        """list: argument for the exogenous / independent variable(s)"""
        return self._X_list

    @property
    def cov_type(self):
        """str: The covariance type.  The names used come from Statsmodels.

        Use of heteroskedasticity-consistent standard errors is recommended as
        common practice in econometrics.

        Examples:
        - 'nonrobust': standard errors not adjusted for heteroskedasticity.
        - 'HC1': robust standard errors (used for 'robust' argument in Stata)
        - 'HC2': robust standard errors obtained via HCCM estimates.
            Recommended by Long & Ervin (1999) where number of observations
            >= 250.
        - 'HC3': robust standard errors obtained via HCCM estimates.
            Recommended by by Long & Ervin (1999) where number of observations
            < 250.
        """
        return self._cov_type

    @property
    def cov_options(self):
        """dict: wrapper for Statsmodels cov_kwds parameter.

        The main difference though is that the dictionary values should not be
        Pandas objects, e.g. df['state'].  They should be column lists
        instead.
        """
        return self._cov_options

    @property
    def alpha(self):
        """Float: the significance level used for confidence intervals and
        p-values in the model summary."""
        return self._alpha

    @property
    def y_standardized(self):
        """pd.Series: endogenous variable standardized (only the values
        used in the model)"""
        return self._y_standardized

    @property
    def X_standardized(self):
        """pd.DataFrame: exogenous variables standardized (only the values
        used in the model)"""
        return self._X_standardized

    # MODEL OUTPUTS & STATES
    @property
    def results(self):
        """statsmodels.regression.linear_model.RegressionResultsWrapper object
        The object contains many details on the fit of the regression model.
        There are dozens of attributes that store such information.

        For a neater summary of the model, use these class attributes:
        - results_output: the object returned by results.summary()
        - model_selection_stats: an assortment of measures contained in
            results, which are used commonly for model selection
            (e.g. AIC, R-squared)
        """
        return self._results

    @property
    def results_output(self):
        """statsmodels.iolib.summary.Summary: tabulated summary of the model.
        The output is well-suited for printing.

        This can all be accessed without having to fit the model explicitly!

        If the model was not fitted before an attempt to access the attribute,
        then the model will be fitted and the results are returned.
        """
        return self._results_output

    @property
    def results_output_standardized(self):
        """pandas.io.formats.style.Styler: tabulated summary of the
        unstandardized and standardized estimates from the regression.

        For Stata users this method is similar to the listcoef command.

        Columns:
        - coef: raw coefficient (before standardization)
        - t / z: test statistic for the test of whether the estimated coef
            is different from 0
        - P>|t| or P>|z|: p-value for the test statistic
        - coef_stdX: x-standardized coefficient
            e.g. if +7.23, then an increase in x by one standard deviation
            is associated with an increase in y by 7.23 units
        - coef_stdXy: fully standardized coefficient
            e.g. if +0.47, then an increase in x by one standard deviation
            is associated with an increase in y by 0.47 standard deviations
        - stdev_X: standard deviation of x

        NOTE: The object cannot be printed directly, as the formatting of
        the object is done via Pandas.
        Access the `data` attribute from the object if more precise info
        is needed.
        Other attributes can also be accessed for exporting the data.
        """
        return self._results_output_standardized

    @property
    def resid(self):
        return self._resid

    @property
    def resid_standardized(self):
        """pd.Series: standardized residuals obtained from the fitted model."""
        return self._resid_standardized

    @property
    def model_selection_stats(self):
        """dict: model selection stats (keys) and their values from the model.

        Examples of stats include Root MSE, AIC, BIC, R-squared,
        R-squared (adjusted).
        """
        return self._model_selection_stats

    @property
    def is_fitted(self):
        "Boolean: indicator for whether the model has been fitted."
        return self._is_fitted

    def fit(self, *, printing=False):
        """Fit the model and save the results in the model object.

        Ensure the model dataset does not contain NaN values, inf
        values, categorical or string data.

        Args:
            printing (bool, optional): display print statements to show
                progress of function calls (e.g. 'Model fitted'). Defaults
                to False.

        Raises:
            ValueError: remove +/- inf values from the model dataset.
            TypeError: encode categorical columns as dummies before fitting
                model.
            ValueError: remove NaN values from the model dataset.

        Returns:
            Instance of the WLS model object, with the model estimates
            now stored as attributes.
        """
        _df_input_conditions(self._X, self._y)

        model = sm.WLS(self._y, sm.add_constant(self._X),
                       weights=self._w, has_const=True)

        if printing:
            print("Model fitting in progress...")
        if self._cov_options:
            self._results = model.fit(cov_type=self._cov_type,
                                      cov_kwds=self._get_cov_kwds())
        else:
            self._results = model.fit(cov_type=self._cov_type)

        self._results_output = self._results.summary(alpha=self._alpha)
        self._resid = self._results.resid

        model_selection_dict = {"root_mse": np.sqrt(self._results.mse_resid),
                                "r_squared": self._results.rsquared,
                                "r_squared_adj": self._results.rsquared_adj,
                                "aic": self._results.aic,
                                "bic": self._results.bic}
        self._model_selection_stats = model_selection_dict

        self._standardize_results()

        self._is_fitted = True
        if printing:
            print("Model fitted.")

        return self

    def _standardize_results(self):
        """Take the unstandardized model and make its results standardized.

        Pipeline:
        - Drop any rows with NaNs (done in regress function)
        - Standardization of X and y
        - Fit model on standardized X and y
        - Gather relevant estimates in a Pandas DataFrame & set to attribute
        """
        # Standardization accounts for NaN values (via Pandas)
        w_stats_tuple = self._get_weighted_stats(self._X, self._y,
                                                 self._w)
        mean_Xw, mean_yw = w_stats_tuple[0], w_stats_tuple[1]
        stdev_X, stdev_y = w_stats_tuple[2], w_stats_tuple[3]
        # Standard error for weighted X vars:
        Xw_mean_se = (self._X - mean_Xw) / stdev_X
        # Standard error for weighted y:
        yw_mean_se = (self._y - mean_yw) / stdev_y

        # Model fitting
        model_standardized = sm.WLS(yw_mean_se, sm.add_constant(Xw_mean_se),
                                    weights=self._w)
        results_obj = model_standardized.fit(cov_type=self._cov_type)

        # Initialize dataframe (regressors in index only)
        output_indices = results_obj.params.drop('const').index
        output_cols = ['coef', 't', 'P>|t|',
                       'coef_stdX', 'coef_stdXy', 'stdev_X']
        std_results_output = (pd.DataFrame(index=output_indices,
                                           columns=output_cols)
                              .rename_axis(self._y.name))

        # Gather values from model that took the raw data
        std_results_output['coef'] = self._results.params
        std_results_output['t'] = self._results.tvalues  # col 1
        std_results_output['P>|t|'] = self._results.pvalues  # col 2
        if not results_obj.use_t:
            # Output will be labelled as z-scores, not t-values
            std_results_output.rename(columns={'t': 'z', 'P>|t|': 'P>|z|'},
                                      inplace=True)
        test_dist_name = std_results_output.columns[1]  # store for dict later
        p_col_name = std_results_output.columns[2]  # store for dict later
        # Gather values from the model that took the standardized data
        std_results_output['coef_stdXy'] = results_obj.params
        std_results_output['coef_stdX'] = results_obj.params * stdev_y
        std_results_output['stdev_X'] = stdev_X

        # Make Pandas Styler object
        std_results_output = std_results_output\
            .style.format({'coef': "{:+.4f}",
                           test_dist_name: '{:+.3f}',
                           p_col_name: '{:.3f}',
                           'coef_stdX': '{:+.4f}',
                           'coef_stdXy': '{:+.4f}',
                           'stdev_X': '{:.4f}'})
        std_results_output.set_caption(
            "Unstandardized and Standardized Estimates")
        self._results_output_standardized = std_results_output
        pass

    def _get_weighted_stats(self, X, y, weights):
        """Gets the weighted mean and standard deviation for each variable
        in X and y, based on an array of weights."""
        Xw_stat_obj = DescrStatsW(self._X, weights=self._w, ddof=1)

        # Weighted standard deviation for X vars:
        std_Xw = np.sqrt(np.abs(Xw_stat_obj.var_ddof(1)))  # abs for w_sum <1
        mean_Xw = Xw_stat_obj.mean  # Numpy array shape: (regressors, )

        yw_stat_obj = DescrStatsW(self._y, weights=self._w, ddof=1)
        # Weighted standard deviation for y:
        std_yw = np.sqrt(np.abs(yw_stat_obj.var_ddof(1)))  # abs for w_sum <1
        mean_yw = yw_stat_obj.mean  # Numpy array shape: (regressors, )

        return mean_Xw, mean_yw, std_Xw, std_yw

    def predict(self, X_predict, *, within_sample=True):
        """Predict the value(s) of given example(s) based on the fitted model.

        The prediction for an example will return as NaN if:
            1) There is a NaN value in any of the regressor values.
            2) within_sample is True and there is a regressor value outside
                the sample.

        Args:
            X_predict (Numpy array): values of the regressors, with shape
                (# examples, # regressors)
            within_sample (bool, optional): Defaults to True.  If a regressor
                has a value outside of the data used to fit the data, then
                NaN value is predicted.

        Raises:
            AssertionError: Model needs to be fitted before prediction.
            ValueError: Check that X_predict is of shape
                (# examples, # regressors)

        Returns:
            np.ndarray: shape (# examples, ) with a prediction for
            each example.
        """
        if not self._is_fitted:
            raise ValueError("Ensure model is fitted first.")

        regressors_count = self._X.shape[1]

        if type(X_predict) != np.ndarray:
            raise TypeError(
                "Pass X_predict as Numpy array with shape (# examples, # regressors).")
        examples_to_predict = X_predict.shape[0]
        regressors_detected = X_predict.shape[1]

        if regressors_detected != regressors_count:
            raise ValueError(
                """Check that X_predict shape corresponds with the number
                of regressors.""")

        with np.errstate(invalid='ignore'):
            # ^ Error suppressed for cases where X_predict contains NaN
            # If there is a NaN, then the Numpy comparison still leads
            # to a NaN prediction
            # Truth array `vals_in_range` shape (# examples, # regressors)
            vals_in_range_min = np.less_equal(np.tile(self._X.min().T, (examples_to_predict, 1)),
                                              X_predict)
            vals_in_range_max = np.greater_equal(X_predict,
                                                 np.tile(self._X.min().T, (examples_to_predict, 1)))
            vals_in_range = vals_in_range_min & vals_in_range_max
            # Truth array - shape (# examples, )
            # for whether each observation has all X vals in range
            all_vals_in_range = vals_in_range.all(axis=1)

            # Add constant into prediction
            # Make X_predict have shape (# examples, # regressors + 1)
            X_predict = np.concatenate([np.ones((examples_to_predict, 1)),
                                        X_predict], axis=1)

        # Statsmodels predict takes arg w/ shape (# examples, # regressors + 1)
        preds = self._results.predict(exog=X_predict)
        if within_sample:
            preds = np.where(all_vals_in_range, preds, np.NaN)
        return preds

    def significant_regressors(self, alpha):
        """Return a list of significant regressors from the regression.

        Args:
            alpha (float): The specified significance level - in range (0, 0.1]

        Raises:
            AssertionError: ensure that the model is fitted.
            TypeError: ensure that alpha is a float.
            ValueError: ensure that alpha is in range (0, 0.1]

        Returns:
            list: a list of the significant regressor names, if any.
                If no regressors are significant, then None is returned.
        """
        if not self._is_fitted:
            raise ValueError("Ensure model is fitted first.")

        if type(alpha) is not float:
            raise TypeError(
                "Ensure that alpha is a float number in range (0, 0.1]")

        if alpha <= 0 or alpha > 0.1:
            raise ValueError(
                "Ensure significance level is a float number in range (0, 0.1]")

        regressor_pvalues = (self._results.pvalues
                             .drop('const'))  # Pandas Series

        # Find the integer indices of the significant regressors
        indices_significant = np.where(regressor_pvalues <= alpha)[0]
        # Return the names of significant regressors as a list
        if indices_significant.size == 0:
            return []
        else:
            return regressor_pvalues.iloc[indices_significant].index.to_list()

    def diagnostic_plot(self, plot_name, *, ax=None,
                        predictor=None):
        """Return a regression diagnostic plot.

        Recommended code block for plotting:
            fig, ax = plt.subplots()
            model1.diagnostic_plot('pp_plot', ax=ax)
            plt.show()
        Without plt.show(), the P-P and Q-Q plots will display twice in
        Jupyter notebook due to how the functions are coded in Statsmodels.


        Args:
            plot_name (str): A regression diagnostic plot from:
                - 'pp_plot': P-P plot
                - 'qq_plot': Q-Q plot
                - 'rvf_plot': plot of residuals against fitted values.
                - 'rvp_plot': plot of residuals against values of a predictor
                    (note: 'predictor' keyword argument must be specified).
            ax (Axes, optional): Defaults to None.  An Axes argument
                to use for plotting.
            predictor (str): Defaults to None - required only when calling
                up an 'rvp_plot'.  Specify a regressor for an 'rvp_plot'.

        Returns:
            Figure: the plot as a Matplotlib Figure object.
        """
        if not self._is_fitted:
            raise ValueError("Ensure model is fitted first.")

        if plot_name not in ['pp_plot', 'qq_plot', 'rvf_plot', 'rvp_plot']:
            raise ValueError(
                "Ensure that a valid plot_name is passed to the method.")
        if plot_name == 'rvp_plot' and not predictor:
            raise ValueError("Ensure that a regressor is specified when calling rvp_plot.")
        if plot_name == 'rvp_plot' and predictor not in self.X_list:
            raise ValueError("Ensure that the regressor is in X_list.")

        if ax is None:
            ax = plt.gca()

        if plot_name == 'pp_plot':
            fig = pp_plot(self.results.resid, ax=ax)
        if plot_name == 'qq_plot':
            fig = qq_plot(self.results.resid, ax=ax)
        if plot_name == 'rvf_plot':
            fig = plot_residuals_vs_fitted_values(
                self.results.resid, self.results.fittedvalues, ax=ax)
        if plot_name == 'rvp_plot':
            fig = plot_residuals_vs_predictor_values(
                self, predictor=predictor, ax=ax)
        return fig


class OLS(WLS):
    """Ordinary Least Squares (OLS) model.

    Regression pipeline:
        - Initialise model object (specify source dataframe and X & y
            columns)
        - Fit model:
            - Run model and store estimates as model object attributes
            - Run model on standardized X and y and store estimates in
                attributes.

    Args:
        df (pd.DataFrame): Pandas DataFrame that contains the data to use
            for modelling.  Each row is an observation and each column is
            either an independent variable (regressor / exogenous
            variable) or a dependent variable (endogenous variable).
        y_list (list): list containing the dependent variable,
            e.g. ['points']
        X_list (list): list containing one or more regressors,
            e.g. ['exper', 'age', 'coll', 'expersq']
        cov_type (str, optional): Defaults to 'nonrobust'.  Standard errors
            type.
            - 'nonrobust': standard errors not adjusted for
                heteroskedasticity.
            - 'HC1': robust standard errors (used for 'robust' argument in
                Stata)
            - 'HC2': robust standard errors obtained via HCCM estimates.
                Recommended by Long & Ervin (1999) where number of
                observations >= 250.
            - 'HC3': robust standard errors obtained via HCCM estimates.
                Recommended by by Long & Ervin (1999) where number of
                observations < 250.
        cov_options (dict, optional): Specify keyword arguments for cov_type.
            This is a wrapper around the cov_kwds parameter in Statsmodels,
            although column lists are used for dict values instead of Pandas
            objects.
            e.g. when specifying a group 'state' for clustered standard errors,
            the form is cov_options={'groups': ['state']},
            instead of cov_kwds={'groups': df['state']}.
        alpha (float, optional): Defaults to 0.05.  The significance level
            used for reporting confidence intervals in the model summary.

    Methods:
        fit: fit the model and store the results in the model object.
        diagnostic_plot: create a plot for regression diagnostics,
            e.g. P-P plot, Q-Q plot, residuals vs predicted values plot,
            residuals vs fitted values plot.
        predict: predict the value(s) of given example(s) based on the fitted
            model.
        significant_regressors: return a list of regressors that are
            significant for a given significance level (alpha).

    Attributes (main):
        model_selection_stats (dict): Model selection stats such as Root MSE,
            AIC, BIC, R-squared and R-squared (adjusted) stored in one place.
        results_output (Statsmodels Summary): Summary object that displays the
            main results from the model.  Suitable for printing in various
            formats.
        results_output_standardized (pd Styler): Summary object that
            displays the main results from the model, in the form of
            standardized estimates.  NOT suitable for print statements.
        results (Statsmodels object): stores information from the Statsmodels
            OLS regression.
        resid (pd.Series): residuals obtained from the fitted model.
        resid_standardized (pd.Series): standardized residuals obtained
            from the fitted model.
        y (pd.Series): contains the values of the independent variable
            for the observations that were used to fit the model.
        y_standardized (pd.Series): y in the form of standardized
            estimates.
        X (pd.DataFrame): contains the values of the regressors for the
            observations used to fit the model.
        X_standardized (pd.DataFrame): X in the form of
            standardized estimates.
        is_fitted (Boolean): indicator for whether the model has been fitted.
    Attributes (auxiliary - used to store arguments):
        df
        cov_type
        cov_options
        alpha
        w
    """

    def __init__(self, df, y_list, X_list, *,
                 cov_type='nonrobust', cov_options=None, alpha=0.05):
        """Initializes the OLS model object."""
        # Model inputs (attributes from arguments):
        self._df = df
        [y_name] = y_list  # sequence unpacking in order to make Series
        self._y = df[y_name]  # Pandas Series
        if len(X_list) == 1:
            [x_name] = X_list
            self._X = df[x_name].to_frame()  # Pandas dataframe
        else:
            self._X = df[X_list]  # Pandas dataframe
        self._y_list, self._X_list = y_list, X_list
        self._w = pd.Series(np.ones(len(self._X)))
        self._cov_type = cov_type
        self._cov_options = cov_options if cov_options else {}
        self._alpha = alpha
        self._is_fitted = False

    def fit(self, *, printing=False):
        """Fit the model and save the results in the model object.

        Ensure the model dataset does not contain NaN values, inf
        values, categorical or string data.

        Args:
            printing (bool, optional): display print statements to show
                progress of function calls (e.g. 'Model fitted'). Defaults
                to False.

        Raises:
            ValueError: remove +/- inf values from the model dataset.
            TypeError: encode categorical columns as dummies before fitting
                model.
            ValueError: remove NaN values from the model dataset.

        Returns:
            Instance of the OLS model object, with the model estimates
            now stored as attributes.
        """
        _df_input_conditions(self._X, self._y)

        model = sm.OLS(self._y, sm.add_constant(self._X))

        if printing:
            print("Model fitting in progress...")
        if self._cov_options:
            self._results = model.fit(cov_type=self._cov_type,
                                      cov_kwds=self._get_cov_kwds())
        else:
            self._results = model.fit(cov_type=self._cov_type)

        self._results_output = self._results.summary(alpha=self._alpha)
        self._resid = self._results.resid

        model_selection_dict = {"root_mse": np.sqrt(self._results.mse_resid),
                                "r_squared": self._results.rsquared,
                                "r_squared_adj": self._results.rsquared_adj,
                                "aic": self._results.aic,
                                "bic": self._results.bic}
        self._model_selection_stats = model_selection_dict

        self._standardize_results()

        self._is_fitted = True
        if printing:
            print("Model fitted.")

        return self

    def _standardize_results(self):
        """Take the unstandardized model and make its results standardized.

        Pipeline:
        - Drop any rows with NaNs (done in regress function)
        - Standardization of X and y
        - Fit model on standardized X and y
        - Gather relevant estimates in a Pandas DataFrame & set to attribute
        """
        # Standardization accounts for NaN values (via Pandas)
        stdev_X, stdev_y = self._X.std(ddof=1), self._y.std(ddof=1)
        self._X_standardized = (
            self._X - self._X.mean()) / stdev_X
        self._y_standardized = (
            self._y - self._y.mean()) / stdev_y

        # Model fitting
        model_standardized = sm.OLS(self._y_standardized,
                                    sm.add_constant(self._X_standardized))
        if self._cov_options:
            results_obj = model_standardized.fit(cov_type=self._cov_type,
                                                 cov_kwds=self._get_cov_kwds())
        else:
            results_obj = model_standardized.fit(cov_type=self._cov_type)

        self._resid_standardized = pd.Series((self._results.get_influence()
                                              .resid_studentized_internal),
                                             index=self._resid.index,
                                             name='resid_standardized')

        # Initialize dataframe (regressors in index only)
        output_indices = results_obj.params.drop('const').index
        output_cols = ['coef', 't', 'P>|t|',
                       'coef_stdX', 'coef_stdXy', 'stdev_X']
        std_results_output = (pd.DataFrame(index=output_indices,
                                           columns=output_cols)
                              .rename_axis(self._y.name))

        # Gather values from model that took the raw data
        std_results_output['coef'] = self._results.params
        std_results_output['t'] = self._results.tvalues  # col 1
        std_results_output['P>|t|'] = self._results.pvalues  # col 2
        if not results_obj.use_t:
            # Output will be labelled as z-scores, not t-values
            std_results_output.rename(columns={'t': 'z', 'P>|t|': 'P>|z|'},
                                      inplace=True)
        test_dist_name = std_results_output.columns[1]  # store for dict later
        p_col_name = std_results_output.columns[2]  # store for dict later
        # Gather values from the model that took the standardized data
        std_results_output['coef_stdXy'] = results_obj.params
        std_results_output['coef_stdX'] = results_obj.params * stdev_y
        std_results_output['stdev_X'] = stdev_X

        # Make Pandas Styler object
        std_results_output = std_results_output\
            .style.format({'coef': "{:+.4f}",
                           test_dist_name: '{:+.3f}',
                           p_col_name: '{:.3f}',
                           'coef_stdX': '{:+.4f}',
                           'coef_stdXy': '{:+.4f}',
                           'stdev_X': '{:.4f}'})
        std_results_output.set_caption(
            "Unstandardized and Standardized Estimates")
        self._results_output_standardized = std_results_output
        pass

    def _get_cov_kwds(self):
        # Appelpy cov_options -> Statsmodels cov_kwds
        cov_kwds = self._cov_options.copy()

        # 'groups' list -> Pandas dataframe/series
        if 'groups' in self._cov_options:
            cov_kwds['groups'] = (self._df.loc[:, self._cov_options['groups']]
                                  .copy())

        # 'time' list -> Numpy array
        if 'time' in self._cov_options:
            cov_kwds['time'] = (self._df.loc[:, self._cov_options['time']]
                                .to_numpy().squeeze())

        return cov_kwds
