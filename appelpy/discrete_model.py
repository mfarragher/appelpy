from .utils import _SuppressPrints, _df_input_conditions
import numpy as np
import pandas as pd
import statsmodels.api as sm

__all__ = ['Logit']


class Logit:
    """Logistic regression model.

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
        alpha (float, optional): Defaults to 0.05.  The significance level
            used for reporting confidence intervals in the model summary.
        printing (Bool, optional): display print statements for function calls.
            Defaults to True.
    Methods:
        predict: predict the probability value(s) of given example(s) based
            on the fitted model.
        significant_regressors: return a list of regressors that are
            significant for a given significance level (alpha).

    Attributes (main):
        model_selection_stats (dict): Model selection stats such as
            log-likelihood in one place.
        results_output (Statsmodels Summary): Summary object that displays the
            main results from the model.  Suitable for printing in various
            formats.
        results_output_standardized (pd Styler): Summary object that
            displays the main results from the model, in the form of
            standardized estimates.  NOT suitable for print statements.
        odds_ratios (pd.Series): odds ratio for each regressor in the model.
        log_likelihood (float): value of the log-likelihood for the model.
        results (Statsmodels object): stores information from the Statsmodels
            Logit regression.
        resid (pd.Series): residuals obtained from the fitted model.
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
        alpha
    """

    def __init__(self, df, y_list, X_list, *,
                 alpha=0.05, printing=True):
        """Initializes the Logit model object."""
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
        self._alpha = alpha
        self._is_fitted = False

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
    def y_list(self):
        """list: argument for the endogenous / dependent variable"""
        return self._y_list

    @property
    def X_list(self):
        """list: argument for the exogenous / independent variable(s)"""
        return self._X_list

    @property
    def X_standardized(self):
        """pd.DataFrame: exogenous variables standardized (only the values
        used in the model)"""
        return self._X_standardized

    @property
    def alpha(self):
        """Float: the significance level used for confidence intervals and
        p-values in the model summary."""
        return self._alpha

    # MODEL OUTPUTS & STATES
    @property
    def results(self):
        """statsmodels.discrete.discrete_model.LogitResults object.
        The object contains many details on the fit of the regression model.
        There are dozens of attributes that store such information.

        For a neater summary of the model, use these class attributes:
        - results_output: the object returned by results.summary()
        - model_selection_stats: an assortment of measures contained in
            results, which are used commonly for model selection
            (e.g. log-likelihood)
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

        The standardized coefficients for logistic regression are calculated
        via Long's method.

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
    def model_selection_stats(self):
        """dict: model selection stats (keys) and their values from the model.

        Examples of stats include log likelihood, pseudo R-squared, etc.
        """
        return self._model_selection_stats

    @property
    def odds_ratios(self):
        """pd.Series: the odds ratio for each regressor in the model.
        """
        return self._odds_ratios

    @property
    def log_likelihood(self):
        """float: the log-likelihood statistics for the fitted model.
        """
        return self._log_likelihood

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
            Instance of the Logit model object, with the model estimates
            now stored as attributes.
        """
        _df_input_conditions(self._X, self._y)

        model = sm.Logit(self._y, sm.add_constant(self._X))

        if printing:
            print("Model fitting in progress...")
        with _SuppressPrints():  # hide Statsmodels printing
            self._results = model.fit()
        self._results_output = self._results.summary(alpha=self._alpha)

        model_selection_dict = {"log_likelihood": self._results.llf,
                                "r_squared_pseudo": self._results.prsquared,
                                "aic": self._results.aic,
                                "bic": self._results.bic}
        self._model_selection_stats = model_selection_dict
        self._log_likelihood = self._results.llf
        self._odds_ratios = pd.Series(np.exp(self._results.params
                                             .drop('const')),
                                      name='odds_ratios')

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
        stdev_X = self._X.std(ddof=1)
        self._X_standardized = (
            self._X - self._X.mean()) / stdev_X

        # Model fitting
        model_standardized = sm.Logit(self._y,
                                      sm.add_constant(self._X_standardized))
        results_obj = model_standardized.fit(disp=0)  # hide second output info

        # Initialize dataframe (regressors in index only)
        output_indices = results_obj.params.drop('const').index
        output_cols = ['coef', 't', 'P>|t|',
                       'coef_stdX', 'coef_stdXy', 'stdev_X']
        std_results_output = pd.DataFrame(index=output_indices,
                                          columns=output_cols)
        std_results_output = std_results_output.rename_axis(self._y.name)

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
        std_results_output['coef_stdX'] = results_obj.params
        std_results_output['stdev_X'] = stdev_X

        # Now calculate std_XY (via Long's method):
        var_explained = self._results.fittedvalues.std() ** 2
        var_ystar = var_explained + np.pi ** 2 / 3  # ystar is latent variable
        std_results_output['coef_stdXy'] = ((std_results_output['coef'] *
                                             stdev_X)
                                            / np.sqrt(var_ystar))

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
            vals_in_range_min = np.less_equal(np.tile(self._X.min().T,
                                                      (examples_to_predict, 1)),
                                              X_predict)
            vals_in_range_max = np.greater_equal(X_predict,
                                                 np.tile(self._X.min().T,
                                                         (examples_to_predict, 1)))
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
