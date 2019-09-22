import pytest
import statsmodels.api as sm
import pandas as pd
import numpy as np
from pandas.util.testing import assert_series_equal
from appelpy.discrete_model import Logit


@pytest.fixture(scope='module')
def model_wells():
    df = sm.datasets.get_rdataset('Wells', 'carData').data
    # Pre-processing
    for col in ['switch', 'association']:
        df[col] = np.where(df[col] == 'yes', 1, 0)
    # Model test case
    X_list = ['arsenic', 'distance', 'education', 'association']
    model = Logit(df, ['switch'], X_list)
    return model


def test_aic(model_wells):
    expected_aic = 3917.8
    assert np.round(
        model_wells.model_selection_stats['AIC'], 1) == expected_aic


def test_log_likelihood(model_wells):
    expected_log_likelihood = -1953.913
    assert np.round(model_wells.log_likelihood, 3) == expected_log_likelihood


def test_coefficients(model_wells):
    expected_coef = pd.Series({'const': -0.156712,
                               'arsenic': 0.467022,
                               'distance': -0.008961,
                               'education': 0.042447,
                               'association': -0.124300})
    assert_series_equal(model_wells.results.params.round(6), expected_coef)


def test_standard_errors(model_wells):
    expected_se = pd.Series({'const': 0.099601,
                             'arsenic': 0.041602,
                             'distance': 0.001046,
                             'education': 0.009588,
                             'association': 0.076966})
    assert_series_equal(model_wells.results.bse.round(6), expected_se)


def test_z_scores(model_wells):
    expected_z_score = pd.Series({'const': -1.573,
                                  'arsenic': 11.226,
                                  'distance': -8.569,
                                  'education': 4.427,
                                  'association': -1.615})
    assert_series_equal(model_wells.results.tvalues.round(3), expected_z_score)
    assert not model_wells.results.use_t


def test_significant_regressors(model_wells):
    expected_regressors = ['arsenic', 'distance', 'education']
    assert model_wells.significant_regressors(0.001) == expected_regressors
