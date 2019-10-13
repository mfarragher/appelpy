import pytest
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pandas.util.testing import assert_frame_equal
from appelpy.utils import DummyEncoder


@pytest.fixture(scope='module')
def df_cars93():
    # Load data and pre-processing
    df = sm.datasets.get_rdataset('Cars93', 'MASS').data
    df.columns = (df.columns
                  .str.replace(r"[ ,.,-]", '_')
                  .str.lower())
    return df


def test_bad_arguments(df_cars93):
    with pytest.raises(ValueError):
        dummy_enc = DummyEncoder(df_cars93, separator='#')
    with pytest.raises(ValueError):
        dummy_enc = DummyEncoder(df_cars93)
        dummy_enc.encode({'drivetrain': '4WD'}, nan_policy='naan_bread')


def test_row_of_zero(df_cars93):
    df = df_cars93.copy()

    dummy_enc = DummyEncoder(df)
    df = dummy_enc.encode({'drivetrain': min, 'origin': max})

    assert_frame_equal(dummy_enc.df, df_cars93)

    assert dummy_enc.separator == '_'
    assert dummy_enc.nan_policy == 'row_of_zero'

    expected_base_levels = {'drivetrain': '4WD', 'origin': 'non-USA'}
    expected_new_cols = ['drivetrain_Front', 'drivetrain_Rear', 'origin_USA']

    # column generation check
    assert set(expected_new_cols).issubset(df.columns)
    assert not set(['origin', 'drivetrain']).issubset(df.columns)
    assert dummy_enc.categorical_col_base_levels == expected_base_levels

    # nan policy check
    assert df[expected_new_cols].isna().sum().sum() == 0


def test_dummy_for_nan(df_cars93):
    df = df_cars93.copy()
    df.loc[0, 'drivetrain'] = np.NaN

    dummy_enc = DummyEncoder(df)

    df = dummy_enc.encode({'drivetrain': '4WD'},
                          nan_policy='dummy_for_nan')
    expected_new_cols = ['drivetrain_Front',
                         'drivetrain_Rear', 'drivetrain_nan']

    # column generation check
    assert set(expected_new_cols).issubset(df.columns)

    # nan policy check
    assert df[expected_new_cols].isna().sum().sum() == 0

    # Edge case: if no NaN, then do the row_of_zero behaviour:
    no_nan_df = df_cars93.copy()
    no_nan_dummy_enc = DummyEncoder(no_nan_df)
    no_nan_df = no_nan_dummy_enc.encode({'drivetrain': '4WD'},
                                        nan_policy='dummy_for_nan')
    no_nan_expected_new_cols = ['drivetrain_Front', 'drivetrain_Rear']
    assert set(no_nan_expected_new_cols).issubset(no_nan_df.columns)
    assert no_nan_df[expected_new_cols].isna().sum().sum() == 0


def test_row_of_nan(df_cars93):
    df = df_cars93.copy()
    df.loc[0, 'drivetrain'] = np.NaN

    dummy_enc = DummyEncoder(df)

    df = dummy_enc.encode({'drivetrain': '4WD'}, nan_policy='row_of_nan')
    expected_new_cols = ['drivetrain_Front', 'drivetrain_Rear']

    # column generation check
    assert set(expected_new_cols).issubset(df.columns)

    # nan policy check
    assert (df[expected_new_cols].isna().sum().sum()
            == 1 * len(expected_new_cols))  # 1 nan val per new dummy col
