import pytest
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pandas.util.testing import assert_series_equal
from appelpy.utils import InteractionEncoder
from appelpy.discrete_model import Logit


@pytest.fixture(scope='module')
def df_wells():
    df = sm.datasets.get_rdataset('Wells', 'carData').data
    # Pre-processing
    for col in ['switch', 'association']:
        df[col] = np.where(df[col] == 'yes', 1, 0)
    return df


def test_encoding(df_wells):
    # Interaction feature (continuous * Boolean)
    intn_encoder = InteractionEncoder(df_wells)
    df_test = intn_encoder.encode({'distance': ['arsenic']})

    assert set(df_test.columns) - set(df_wells.columns) == {'distance#arsenic'}
    assert pd.api.types.is_float_dtype(df_test['distance#arsenic'])


def test_model(df_wells):
    df_test = df_wells.copy()
    df_test['distance100'] = df_test['distance'] / 100
    del df_test['distance']
    # Interaction feature
    intn_encoder = InteractionEncoder(df_test)
    df_test = intn_encoder.encode({'distance100': ['arsenic']})

    # Model test case
    X_list = ['arsenic', 'distance100', 'distance100#arsenic']
    model = Logit(df_test, ['switch'], X_list)
    expected_coef = pd.Series({'const': -0.15,
                               'arsenic': 0.56,
                               'distance100': -0.58,
                               'distance100#arsenic': -0.18})
    assert_series_equal(model.results.params.round(2), expected_coef)
    expected_se = pd.Series({'const': 0.12,
                             'arsenic': 0.07,
                             'distance100': 0.21,
                             'distance100#arsenic': 0.10})
    assert_series_equal(model.results.bse.round(2), expected_se)
