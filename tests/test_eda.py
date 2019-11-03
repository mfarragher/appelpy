import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.figure
import matplotlib.pyplot as plt
import pytest
from pandas.util.testing import assert_series_equal, assert_numpy_array_equal
from appelpy.eda import statistical_moments, correlation_heatmap


@pytest.fixture(scope='module')
def df_caschool():
    df = sm.datasets.get_rdataset('Caschool', 'Ecdat').data

    return df


@pytest.mark.remote_data
def test_statistical_moments(df_caschool):
    moments = statistical_moments(df_caschool, kurtosis_fisher=False)
    moments_default = statistical_moments(df_caschool)

    # Pearson kurtosis (the one used in Stata)
    expected_testscr = pd.Series({'mean': 654.1565,
                                  'var': 363.0301,
                                  'skew': 0.0916151,
                                  'kurtosis': 2.745712},
                                 name='testscr')
    expected_str = pd.Series({'mean': 19.64043,
                              'var': 3.578952,
                              'skew': -0.0253655,
                              'kurtosis': 3.609597},
                             name='str')

    # Fisher kurtosis (Scipy default): subtract 3 from Pearson kurtosis
    expected_testscr_default = pd.Series({'mean': 654.1565,
                                          'var': 363.0301,
                                          'skew': 0.0916151,
                                          'kurtosis': 2.745712 - 3},
                                         name='testscr')
    expected_str_default = pd.Series({'mean': 19.64043,
                                      'var': 3.578952,
                                      'skew': -0.0253655,
                                      'kurtosis': 3.609597 - 3},
                                     name='str')

    assert_series_equal(moments.loc['testscr'], expected_testscr,
                        check_dtype=False)
    assert_series_equal(moments.loc['str'], expected_str,
                        check_dtype=False)
    assert_series_equal(moments_default.loc['testscr'],
                        expected_testscr_default,
                        check_dtype=False)
    assert_series_equal(moments_default.loc['str'], expected_str_default,
                        check_dtype=False)


@pytest.mark.remote_data
def test_correlation_heatmap(df_caschool):
    # Close open figures at start of new test
    plt.cla()
    plt.close('all')

    cols_subset = ['avginc', 'readscr', 'mathscr', 'expnstu']
    df_subset = df_caschool[cols_subset].copy()

    fig = correlation_heatmap(df_subset)
    assert isinstance(fig, matplotlib.figure.Figure)

    # Expected data
    expected_mask = np.array([True,  True,  True,  True,
                              False,  True,  True,  True,
                              False, False,  True,  True,
                              False, False, False,  True])
    expected_data = np.array([1.0000, 0.6978, 0.6994, 0.3145,
                              0.6978, 1.0000, 0.9229, 0.2179,
                              0.6994, 0.9229, 1.0000, 0.1550,
                              0.3145, 0.2179, 0.1550, 1.0000])

    # Actual data
    quad_mesh_array = fig.get_axes()[0].collections[0].get_array()
    assert_numpy_array_equal(np.round(quad_mesh_array.data, 4),
                             expected_data)
    assert_numpy_array_equal(quad_mesh_array.mask,
                             expected_mask)
