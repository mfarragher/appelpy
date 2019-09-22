import pytest
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pandas.util.testing import assert_series_equal
from appelpy.diagnostics import heteroskedasticity_test
from appelpy.utils import DummyEncoder
from appelpy.linear_model import OLS


# - MODELS -

@pytest.fixture(scope='module')
def model_mtcars_final():
    df = sm.datasets.get_rdataset('mtcars').data
    X_list = ['wt', 'qsec', 'am']
    model = OLS(df, ['mpg'], X_list)
    return model


@pytest.fixture(scope='module')
def model_cars():
    df = sm.datasets.get_rdataset('cars').data
    X_list = ['speed']
    model = OLS(df, ['dist'], X_list)
    return model


@pytest.fixture(scope='module')
def model_cars93():
    # Load data and pre-processing
    df = sm.datasets.get_rdataset('Cars93', 'MASS').data
    df.columns = (df.columns
                  .str.replace(r"[ ,.,-]", '_')
                  .str.lower())
    # Dummy columns
    dummy_encoder = DummyEncoder(df, separator='_')
    df = dummy_encoder.encode({'type': 'Compact',
                               'airbags': 'None',
                               'origin': 'USA'})
    df.columns = (df.columns
                  .str.replace(r"[ ,.,-]", '_')
                  .str.lower())
    # Model
    X_list = ['type_large', 'type_midsize', 'type_small',
              'type_sporty', 'type_van',
              'mpg_city',
              'airbags_driver_&_passenger', 'airbags_driver_only',
              'origin_non_usa']
    model = OLS(df, ['price'], X_list)
    return model


# - TESTS -

def test_coefficients(model_mtcars_final):
    expected_coef = pd.Series({'const': 9.6178,
                               'wt': -3.9165,
                               'qsec': 1.2259,
                               'am': 2.9358})
    assert_series_equal(model_mtcars_final.results.params.round(4),
                        expected_coef)


def test_coefficients_beta(model_cars93):
    expected_coefs = pd.Series({'type_large': 0.10338,
                                'type_midsize': 0.22813,
                                'type_small': -0.01227,
                                'type_sporty': 0.01173,
                                'type_van': -0.02683,
                                'mpg_city': -0.46296,
                                'airbags_driver_&_passenger': 0.34998,
                                'airbags_driver_only': 0.23687,
                                'origin_non_usa': 0.26742},
                               name='coef_stdXy')
    expected_coefs.index.name = 'price'
    assert_series_equal((model_cars93
                         .results_output_standardized
                         .data['coef_stdXy']
                         .round(5)),
                        expected_coefs)


def test_standard_errors(model_mtcars_final):
    expected_se = pd.Series({'const': 6.9596,
                             'wt': 0.7112,
                             'qsec': 0.2887,
                             'am': 1.4109})
    assert_series_equal(model_mtcars_final.results.bse.round(4),
                        expected_se)


def test_t_scores(model_mtcars_final):
    expected_t_score = pd.Series({'const': 1.382,
                                  'wt': -5.507,
                                  'qsec': 4.247,
                                  'am': 2.081})
    assert_series_equal(model_mtcars_final.results.tvalues.round(3),
                        expected_t_score)
    assert model_mtcars_final.results.use_t


def test_model_selection_stats(model_mtcars_final):
    expected_root_mse = 2.459
    expected_r_squared = 0.8497
    expected_r_squared_adj = 0.8336

    assert(np.round((model_mtcars_final
                     .model_selection_stats['Root MSE']), 3)
           == expected_root_mse)
    assert(np.round((model_mtcars_final
                     .model_selection_stats['R-squared']), 4)
           == expected_r_squared)
    assert(np.round((model_mtcars_final
                     .model_selection_stats['R-squared (adj)']), 4)
           == expected_r_squared_adj)


def test_significant_regressors(model_mtcars_final):
    expected_regressors_001 = ['wt', 'qsec']  # 0.1% sig
    expected_regressors_01 = ['wt', 'qsec']  # 1% sig
    expected_regressors_05 = ['wt', 'qsec', 'am']  # 5% sig

    assert (model_mtcars_final.significant_regressors(0.001)
            == expected_regressors_001)
    assert (model_mtcars_final.significant_regressors(0.01)
            == expected_regressors_01)
    assert (model_mtcars_final.significant_regressors(0.05)
            == expected_regressors_05)


def test_heteroskedasticity_diagnostics(model_cars):
    expected_lm, expected_pval = (4.650233, 0.03104933)
    lm, pval = heteroskedasticity_test('breusch_pagan', model_cars)
    assert (np.round(lm, 6) == expected_lm)
    assert (np.round(pval, 8) == expected_pval)
