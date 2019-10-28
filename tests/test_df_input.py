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

    # x var
    df.loc[0, 'mpg_city'] = np.NaN
    with pytest.raises(ValueError, match=r'.*NaN*.'):
        model = OLS(df, ['price'], ['mpg_city'])
        model.fit(printing=False)

    # y var
    df = df_cars93.copy()
    df.loc[0, 'price'] = np.NaN
    with pytest.raises(ValueError, match=r'.*NaN*.'):
        model = OLS(df, ['price'], ['mpg_city'])
        model.fit(printing=False)


def test_inf_values(df_cars93):
    df = df_cars93.copy()

    # x var
    df.loc[0, 'mpg_city'] = np.inf
    with pytest.raises(ValueError, match=r'.*infinite*.'):
        model = OLS(df, ['price'], ['mpg_city'])
        model.fit(printing=False)

    # y var
    df = df_cars93.copy()
    df.loc[0, 'price'] = -np.inf
    with pytest.raises(ValueError, match=r'.*infinite*.'):
        model = OLS(df, ['price'], ['mpg_city'])
        model.fit(printing=False)


def test_string_values(df_cars93):
    # x var
    with pytest.raises(TypeError, match=r'.*string*.'):
        model = OLS(df_cars93, ['price'], ['type'])
        model.fit(printing=False)

    # y var
    df = df_cars93.copy()
    df.loc[0, 'price'] = 'bad_idea'
    with pytest.raises(TypeError, match=r'.*string*.'):
        model = OLS(df, ['price'], ['mpg_city'])
        model.fit(printing=False)


def test_category_values(df_cars93):
    df = df_cars93.copy()
    df['type'] = df['type'].astype('category')
    with pytest.raises(TypeError, match=r'.*Category*.'):
        model = OLS(df, ['price'], ['type'])
        model.fit(printing=False)
