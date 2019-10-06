import pytest
import pandas as pd
import statsmodels.api as sm
from pandas.util.testing import assert_series_equal
from appelpy.diagnostics import variance_inflation_factors
from appelpy.linear_model import OLS


@pytest.fixture(scope='module')
def model_mtcars_initial():
    df = sm.datasets.get_rdataset('mtcars').data
    X_list = ['cyl', 'disp', 'hp', 'drat', 'wt',
              'qsec', 'vs', 'am', 'gear', 'carb']
    model = OLS(df, ['mpg'], X_list)
    return model


def test_variance_inflation_factors(model_mtcars_initial):
    expected_vif = pd.Series({'cyl': 15.373833,
                              'disp': 21.620241,
                              'hp': 9.832037,
                              'drat': 3.374620,
                              'wt': 15.164887,
                              'qsec': 7.527958,
                              'vs': 4.965873,
                              'am': 4.648487,
                              'gear': 5.357452,
                              'carb': 7.908747}, name='VIF')
    df_vif = variance_inflation_factors(model_mtcars_initial.X)
    assert_series_equal(df_vif['VIF'].round(6), expected_vif)
