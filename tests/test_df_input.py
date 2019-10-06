import pytest
import pandas as pd
import numpy as np
import statsmodels.api as sm
from appelpy.linear_model import OLS


@pytest.fixture
def df_cars93(scope='module'):
    # Load data and pre-processing
    df = sm.datasets.get_rdataset('Cars93', 'MASS').data
    df.columns = (df.columns
                  .str.replace(r"[ ,.,-]", '_')
                  .str.lower())
    return df


def test_nan_values(df_cars93):
    df = df_cars93.copy()
    df.loc[0, 'mpg_city'] = np.NaN
    with pytest.raises(ValueError, match=r'.*NaN*.'):
        OLS(df, ['price'], ['mpg_city'])


def test_inf_values(df_cars93):
    df = df_cars93.copy()
    df.loc[0, 'mpg_city'] = np.inf
    with pytest.raises(ValueError, match=r'.*infinite*.'):
        OLS(df, ['price'], ['mpg_city'])


def test_string_values(df_cars93):
    with pytest.raises(TypeError, match=r'.*string*.'):
        OLS(df_cars93, ['price'], ['type'])


def test_category_values(df_cars93):
    df = df_cars93.copy()
    df['type'] = df['type'].astype('category')
    with pytest.raises(TypeError, match=r'.*Category*.'):
        OLS(df, ['price'], ['type'])
