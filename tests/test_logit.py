import pytest
import statsmodels
import statsmodels.api as sm
import pandas as pd
import numpy as np
from pandas.util.testing import (assert_series_equal,
                                 assert_frame_equal,
                                 assert_numpy_array_equal)
from appelpy.discrete_model import Logit
from appelpy.utils import DummyEncoder


@pytest.fixture(scope='module')
def model_wells():
    df = sm.datasets.get_rdataset('Wells', 'carData').data
    # Pre-processing
    for col in ['switch', 'association']:
        df[col] = np.where(df[col] == 'yes', 1, 0)
    # Model test case
    X_list = ['arsenic', 'distance', 'education', 'association']
    model = Logit(df, ['switch'], X_list).fit()
    return model


@pytest.fixture(scope='module')
def model_birthwt():
    df = sm.datasets.get_rdataset('birthwt', 'MASS').data

    # Race column
    race_dict = {1: 'white', 2: 'black', 3: 'other'}
    df['race'] = (df['race'].replace(race_dict)
                  .astype('category'))

    dummy_enc = DummyEncoder(df, {'race': 'white'})
    df = dummy_enc.transform()
    for col in ['race_black', 'race_other']:
        df[col] = df[col].astype(int)

    # Model test case
    X_list = ['smoke', 'race_black', 'race_other']
    y_list = ['low']

    model = Logit(df, y_list, X_list).fit()
    return model


@pytest.fixture(scope='module')
def model_mtcars_vs():
    df = sm.datasets.get_rdataset('mtcars').data
    X_list = ['wt', 'disp']
    model = Logit(df, ['vs'], X_list).fit()
    return model


@pytest.fixture(scope='module')
def model_mtcars_am():
    df = sm.datasets.get_rdataset('mtcars').data
    X_list = ['mpg']
    model = Logit(df, ['am'], X_list).fit()
    return model


@pytest.mark.remote_data
def test_model_not_fitted():
    df = sm.datasets.get_rdataset('mtcars').data
    X_list = ['wt', 'disp']
    model = Logit(df, ['vs'], X_list)

    assert not model.is_fitted
    with pytest.raises(ValueError):
        model.significant_regressors(0.05)
    with pytest.raises(ValueError):
        model.predict(model.X.mean())
    assert_frame_equal(df, model.df)


@pytest.mark.remote_data
def test_prints(capsys):
    df = sm.datasets.get_rdataset('mtcars').data
    X_list = ['wt', 'disp']
    Logit(df, ['vs'], X_list).fit(printing=True)

    captured = capsys.readouterr()
    expected_print = "Model fitting in progress...\nModel fitted.\n"
    assert captured.out == expected_print


@pytest.mark.remote_data
def test_coefficients(model_wells):
    assert model_wells.is_fitted

    expected_coef = pd.Series({'const': -0.156712,
                               'arsenic': 0.467022,
                               'distance': -0.008961,
                               'education': 0.042447,
                               'association': -0.124300})
    assert_series_equal(model_wells.results.params.round(6), expected_coef)

    assert isinstance(model_wells.results_output,
                      statsmodels.iolib.summary.Summary)
    assert isinstance(model_wells.results_output_standardized,
                      pd.io.formats.style.Styler)

    # y_list & X_list
    assert model_wells.y_list == ['switch']
    assert model_wells.X_list == ['arsenic', 'distance', 'education',
                                  'association']


@pytest.mark.remote_data
def test_coefficients_beta(model_mtcars_vs):
    expected_coefs = pd.Series({'wt': 1.5913,
                                'disp': -4.2677},
                               name='coef_stdX')
    expected_coefs.index.name = 'vs'
    assert_series_equal((model_mtcars_vs
                         .results_output_standardized
                         .data['coef_stdX']
                         .round(4)),
                        expected_coefs)


@pytest.mark.remote_data
def test_standard_errors(model_wells):
    expected_se = pd.Series({'const': 0.099601,
                             'arsenic': 0.041602,
                             'distance': 0.001046,
                             'education': 0.009588,
                             'association': 0.076966})
    assert_series_equal(model_wells.results.bse.round(6), expected_se)


@pytest.mark.remote_data
def test_z_scores(model_wells):
    expected_z_score = pd.Series({'const': -1.573,
                                  'arsenic': 11.226,
                                  'distance': -8.569,
                                  'education': 4.427,
                                  'association': -1.615})
    assert_series_equal(model_wells.results.tvalues.round(3), expected_z_score)
    assert not model_wells.results.use_t


@pytest.mark.remote_data
def test_aic(model_wells):
    expected_aic = 3917.8
    assert np.round(
        model_wells.model_selection_stats['AIC'], 1) == expected_aic


@pytest.mark.remote_data
def test_log_likelihood(model_wells):
    expected_log_likelihood = -1953.913
    assert np.round(model_wells.log_likelihood, 3) == expected_log_likelihood


@pytest.mark.remote_data
def test_significant_regressors(model_wells, model_mtcars_vs):
    # Results output:
    assert model_wells.alpha == 0.05

    # Significant regressors method - check p=0.001
    expected_regressors_wells = ['arsenic', 'distance', 'education']
    assert (model_wells.significant_regressors(0.001)
            == expected_regressors_wells)
    expected_regressors_mtcars = []
    assert (model_mtcars_vs.significant_regressors(0.001)
            == expected_regressors_mtcars)

    with pytest.raises(TypeError):
        model_wells.significant_regressors('str')

    with pytest.raises(ValueError):
        model_wells.significant_regressors(np.inf)
    with pytest.raises(TypeError):
        model_wells.significant_regressors(0)
    with pytest.raises(TypeError):
        model_wells.significant_regressors(-1)
    with pytest.raises(ValueError):
        model_wells.significant_regressors(0.11)


@pytest.mark.remote_data
def test_odds_ratio(model_birthwt):
    expected_odds_ratios_smoking = 3.052631
    assert(np.round(model_birthwt.odds_ratios.loc['smoke'], 6)
           == expected_odds_ratios_smoking)


@pytest.mark.remote_data
def test_other_properties(model_mtcars_vs):
    assert isinstance(model_mtcars_vs.X, pd.DataFrame)
    assert isinstance(model_mtcars_vs.y, pd.Series)


@pytest.mark.remote_data
def test_predictions(model_mtcars_vs, model_mtcars_am):
    expected_pred = np.array([0.2361081, np.NaN])
    actual_pred = model_mtcars_vs.predict(
        pd.DataFrame(data={'wt': [2.1, -1000000],
                           'disp': [180, -1000000]}).values)
    assert_numpy_array_equal(np.round(actual_pred, 7),
                             expected_pred)

    expected_pred = np.array([0.2361081, 0])
    actual_pred = model_mtcars_vs.predict(
        pd.DataFrame(data={'wt': [2.1, -1000000],
                           'disp': [180, -1000000]}).values,
        within_sample=False)
    assert_numpy_array_equal(np.round(actual_pred, 7),
                             expected_pred)

    expected_pred = np.array([0.1194021, 0.3862832, 0.7450109])
    actual_pred = model_mtcars_am.predict(
        np.array(([[15], [20], [25]])))
    assert_numpy_array_equal(np.round(actual_pred, 7),
                             expected_pred)

    with pytest.raises(TypeError):
        model_mtcars_am.predict(
            pd.DataFrame(np.array(([[15], [20], [25]]))
                         ))
    with pytest.raises(ValueError):
        # One-regressor model with two regressors fed to predict
        model_mtcars_am.predict(
            np.array(([15, 20, 25],
                      [1, 2, 3])))
