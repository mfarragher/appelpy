import pytest
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pandas.util.testing import assert_series_equal, assert_frame_equal
from appelpy.utils import InteractionEncoder
from appelpy.discrete_model import Logit


@pytest.fixture(scope='module')
def df_wells():
    df = sm.datasets.get_rdataset('Wells', 'carData').data
    # Pre-processing
    for col in ['switch', 'association']:
        df[col] = np.where(df[col] == 'yes', 1, 0)
    return df


@pytest.fixture(scope='module')
def df_caschools():
    df = sm.datasets.get_rdataset('Caschool', 'Ecdat').data

    # Features
    df['hi_str'] = np.where(df['str'] >= 20, 1, 0)
    df['hi_elpct'] = np.where(df['elpct'] >= 10, 1, 0)

    return df


@pytest.fixture(scope='module')
def df_birthwt():
    df = sm.datasets.get_rdataset('birthwt', 'MASS').data

    # Race column
    race_dict = {1: 'white', 2: 'black', 3: 'other'}
    df['race'] = (df['race'].replace(race_dict)
                  .astype('category'))

    return df


def test_cont_cont_encoding(df_wells):
    df_test = df_wells.copy()

    # Interaction feature
    intn_encoder = InteractionEncoder(df_wells)
    df_test = intn_encoder.encode({'distance': ['arsenic']})

    expected_columns_added = ['distance#arsenic']
    expected_columns_removed = []
    assert intn_encoder.columns_added == expected_columns_added
    assert intn_encoder.columns_removed == expected_columns_removed
    assert pd.api.types.is_float_dtype(df_test['distance#arsenic'])


def test_bool_bool_encoding(df_caschools):
    df_test = df_caschools.copy()
    df_test['hi_str'] = np.where(df_test['str'] >= 20, 1, 0)
    df_test['hi_elpct'] = np.where(df_test['elpct'] >= 10, 1, 0)

    # Interaction feature (Boolean * Boolean)
    intn_encoder = InteractionEncoder(df_test)
    df_test = intn_encoder.encode({'hi_str': ['hi_elpct']})

    expected_columns_added = ['hi_str#hi_elpct']
    expected_columns_removed = []
    assert intn_encoder.columns_added == expected_columns_added
    assert intn_encoder.columns_removed == expected_columns_removed
    assert pd.api.types.is_integer_dtype(df_test['hi_str#hi_elpct'])


def test_bool_cont_encoding(df_caschools):
    df_test = df_caschools.copy()
    df_test['hi_elpct'] = np.where(df_test['elpct'] >= 10, 1, 0)

    # Interaction feature (Bool * cont)
    intn_encoder = InteractionEncoder(df_test)
    df_test = intn_encoder.encode({'str': ['hi_elpct']})

    expected_columns_added = ['str#hi_elpct']
    expected_columns_removed = []
    assert intn_encoder.columns_added == expected_columns_added
    assert intn_encoder.columns_removed == expected_columns_removed
    assert pd.api.types.is_float_dtype(df_test['str#hi_elpct'])

    # Interaction feature (cont * Bool)
    intn_encoder = InteractionEncoder(df_test)
    df_test = intn_encoder.encode({'hi_elpct': ['str']})

    expected_columns_added = ['hi_elpct#str']
    expected_columns_removed = []
    assert intn_encoder.columns_added == expected_columns_added
    assert intn_encoder.columns_removed == expected_columns_removed
    assert pd.api.types.is_float_dtype(df_test['str#hi_elpct'])


def test_cat_cont_encoding(df_birthwt):
    df_test = df_birthwt.copy()

    # Interaction feature (cat, cont)
    intn_encoder = InteractionEncoder(df_test)
    df_test = intn_encoder.encode({'race': ['age']})

    expected_columns_added = ['race_black', 'race_white', 'race_other',
                              'race_black#age', 'race_white#age',
                              'race_other#age']  # unsorted
    expected_columns_removed = ['race']
    assert sorted(intn_encoder.columns_added) == sorted(expected_columns_added)
    assert sorted(intn_encoder.columns_removed) == sorted(
        expected_columns_removed)
    for col in intn_encoder.columns_added:  # dummies should be int
        assert pd.api.types.is_integer_dtype(df_test[col])

    # Interaction feature (cont, cat)
    df_test = df_birthwt.copy()
    intn_encoder = InteractionEncoder(df_test)
    df_test = intn_encoder.encode({'age': ['race']})

    expected_columns_added = ['race_black', 'race_white', 'race_other',
                              'age#race_black', 'age#race_white',
                              'age#race_other']  # unsorted
    expected_columns_removed = ['race']
    assert sorted(intn_encoder.columns_added) == sorted(expected_columns_added)
    assert sorted(intn_encoder.columns_removed) == sorted(
        expected_columns_removed)
    for col in intn_encoder.columns_added:  # dummies should be int
        assert pd.api.types.is_integer_dtype(df_test[col])


def test_cat_bool_encoding(df_birthwt):
    df_test = df_birthwt.copy()

    # Interaction feature (cat, bool)
    intn_encoder = InteractionEncoder(df_test)
    df_test = intn_encoder.encode({'race': ['smoke']})

    expected_columns_added = ['race_black', 'race_white', 'race_other',
                              'race_black#smoke', 'race_white#smoke',
                              'race_other#smoke']
    expected_columns_removed = ['race']
    assert sorted(intn_encoder.columns_added) == sorted(expected_columns_added)
    assert sorted(intn_encoder.columns_removed) == sorted(
        expected_columns_removed)
    for col in intn_encoder.columns_added:  # dummies should be int
        assert pd.api.types.is_integer_dtype(df_test[col])

    # Interaction feature (bool, cat)
    df_test = df_birthwt.copy()
    intn_encoder = InteractionEncoder(df_test)
    df_test = intn_encoder.encode({'smoke': ['race']})

    expected_columns_added = ['race_black', 'race_white', 'race_other',
                              'smoke#race_black', 'smoke#race_white',
                              'smoke#race_other']
    expected_columns_removed = ['race']
    assert sorted(intn_encoder.columns_added) == sorted(expected_columns_added)
    assert sorted(intn_encoder.columns_removed) == sorted(
        expected_columns_removed)
    for col in intn_encoder.columns_added:  # dummies should be int
        assert pd.api.types.is_integer_dtype(df_test[col])


def test_cat_cat_encoding(df_birthwt):
    df_test = df_birthwt.copy()
    # ptl is discrete var with vals [0, 1, 2, 3]: make it a category
    df_test['ptl'] = df_test['ptl'].astype('category')

    # Interaction feature (cat, cat)
    intn_encoder = InteractionEncoder(df_test)
    df_test = intn_encoder.encode({'race': ['ptl']})
    expected_columns_added = ['race_black', 'race_white', 'race_other',
                              'ptl_0', 'ptl_1', 'ptl_2', 'ptl_3',
                              'race_black#ptl_0', 'race_black#ptl_1',
                              'race_black#ptl_2', 'race_black#ptl_3',
                              'race_white#ptl_0', 'race_white#ptl_1',
                              'race_white#ptl_2', 'race_white#ptl_3',
                              'race_other#ptl_0', 'race_other#ptl_1',
                              'race_other#ptl_2', 'race_other#ptl_3']
    expected_columns_removed = ['race', 'ptl']
    assert sorted(intn_encoder.columns_added) == sorted(expected_columns_added)
    assert sorted(intn_encoder.columns_removed) == sorted(
        expected_columns_removed)
    for col in intn_encoder.columns_added:  # dummies should be int
        assert pd.api.types.is_integer_dtype(df_test[col])


def test_model_wells(df_wells):
    df_test = df_wells.copy()
    df_test['distance100'] = df_test['distance'] / 100
    del df_test['distance']
    # Interaction feature
    intn_encoder = InteractionEncoder(df_test)

    assert_frame_equal(intn_encoder.df, df_test)  # test attribute
    assert intn_encoder.separator == '_'  # test attribute

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
