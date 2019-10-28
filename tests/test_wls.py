import pandas as pd
import numpy as np
import statsmodels
import statsmodels.api as sm
import pytest
from pandas.util.testing import assert_series_equal
from appelpy.utils import DummyEncoder
from appelpy.linear_model import WLS, OLS


@pytest.fixture(scope='module')
def model_salaries():
    # Load data and pre-processing
    df = sm.datasets.get_rdataset('Salaries', 'carData').data
    df.columns = (df.columns
                  .str.replace(r"[ ,.,-]", '_')
                  .str.lower())
    # Make dummy cols
    base_levels = {'rank': 'AsstProf',
                   'discipline': 'A',
                   'sex': 'Female'}
    df = DummyEncoder(df, base_levels).transform()

    y_list = ['salary']
    X_list = ['rank_AssocProf', 'rank_Prof', 'discipline_B',
              'yrs_since_phd', 'yrs_service', 'sex_Male']

    model_ols = OLS(df, y_list, X_list).fit()

    weights = 1 / (model_ols.resid) ** 2

    model_wls = WLS(df, y_list, X_list, w=weights).fit()

    return model_wls


@pytest.fixture(scope='module')
def model_caschools():
    df = sm.datasets.get_rdataset('Caschool', 'Ecdat').data

    # Square income
    df['avginc_sq'] = df['avginc'] ** 2

    # Model: WLS with no weights - equivalent to OLS
    X_list = ['avginc', 'avginc_sq']
    model = WLS(df, ['testscr'], X_list, w=None).fit()
    return model


def test_prints(capsys):
    df = sm.datasets.get_rdataset('Caschool', 'Ecdat').data

    # Square income
    df['avginc_sq'] = df['avginc'] ** 2

    # Model: WLS with no weights - equivalent to OLS
    X_list = ['avginc', 'avginc_sq']

    WLS(df, ['testscr'], X_list, w=None).fit(printing=True)

    captured = capsys.readouterr()
    expected_print = "Model fitting in progress...\nModel fitted.\n"
    assert captured.out == expected_print


def test_coefficients(model_salaries):
    assert model_salaries.is_fitted

    expected_coef = pd.Series({'const': 64897.52,
                               'rank_AssocProf': 13372.32,
                               'rank_Prof': 45544.09,
                               'discipline_B': 14293.17,
                               'yrs_since_phd': 529.08,
                               'yrs_service': -503.16,
                               'sex_Male': 5974.91})
    assert_series_equal(model_salaries.results.params.round(2),
                        expected_coef)

    assert isinstance(model_salaries.results_output,
                      statsmodels.iolib.summary.Summary)
    assert isinstance(model_salaries.results_output_standardized,
                      pd.io.formats.style.Styler)


def test_standard_errors(model_salaries):
    expected_se = pd.Series({'const': 710.12,
                             'rank_AssocProf': 350.33,
                             'rank_Prof': 474.68,
                             'discipline_B': 182.62,
                             'yrs_since_phd': 26.90,
                             'yrs_service': 37.82,
                             'sex_Male': 691.68})
    assert_series_equal(model_salaries.results.bse.round(2),
                        expected_se)

    assert(model_salaries.cov_type == 'nonrobust')


def test_t_scores(model_salaries):
    expected_t_score = pd.Series({'const': 91.390,
                                  'rank_AssocProf': 38.171,
                                  'rank_Prof': 95.947,
                                  'discipline_B': 78.269,
                                  'yrs_since_phd': 19.666,
                                  'yrs_service': -13.303,
                                  'sex_Male': 8.638})
    assert_series_equal(model_salaries.results.tvalues.round(3),
                        expected_t_score)
    assert model_salaries.results.use_t


def test_model_selection_stats(model_salaries):
    expected_root_mse = 1.001
    expected_r_squared = 0.9977
    expected_r_squared_adj = 0.9977

    assert(np.round((model_salaries
                     .model_selection_stats['Root MSE']), 3)
           == expected_root_mse)
    assert(np.round((model_salaries
                     .model_selection_stats['R-squared']), 4)
           == expected_r_squared)
    assert(np.round((model_salaries
                     .model_selection_stats['R-squared (adj)']), 4)
           == expected_r_squared_adj)


def test_significant_regressors(model_salaries):
    expected_regressors_001 = ['rank_AssocProf', 'rank_Prof', 'discipline_B',
                               'yrs_since_phd', 'yrs_service', 'sex_Male']  # 0.1% sig
    expected_regressors_01 = ['rank_AssocProf', 'rank_Prof', 'discipline_B',
                              'yrs_since_phd', 'yrs_service', 'sex_Male']  # 1% sig
    expected_regressors_05 = ['rank_AssocProf', 'rank_Prof', 'discipline_B',
                              'yrs_since_phd', 'yrs_service', 'sex_Male']  # 5% sig

    assert (model_salaries.significant_regressors(0.001)
            == expected_regressors_001)
    assert (model_salaries.significant_regressors(0.01)
            == expected_regressors_01)
    assert (model_salaries.significant_regressors(0.05)
            == expected_regressors_05)

    with pytest.raises(TypeError):
        model_salaries.significant_regressors('str')

    with pytest.raises(ValueError):
        model_salaries.significant_regressors(np.inf)
        model_salaries.significant_regressors(0)
        model_salaries.significant_regressors(-1)
        model_salaries.significant_regressors(0.11)


def test_ols_wls_equivalence(model_caschools):
    # Check weights are vals of 1
    expected_weights = pd.Series(np.ones(len(model_caschools.X)))
    assert_series_equal(model_caschools.w, expected_weights)

    # Attribute check
    assert model_caschools.alpha == 0.05

    # Predictions
    # Est effect of avg 10 -> 11:
    expected_y_hat_diff = 2.9625
    actual_y_hat_diff = (model_caschools.predict(pd.Series([11, 11 ** 2])) -
                         model_caschools.predict(pd.Series([10, 10 ** 2])))[0]
    assert np.round(actual_y_hat_diff, 4) == expected_y_hat_diff

    # Est effect of avg 40 -> 41:
    expected_y_hat_diff = 0.4240
    actual_y_hat_diff = (model_caschools.predict(pd.Series([41, 41 ** 2])) -
                         model_caschools.predict(pd.Series([40, 40 ** 2])))[0]
    assert np.round(actual_y_hat_diff, 4) == expected_y_hat_diff
