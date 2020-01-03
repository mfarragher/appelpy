import pytest
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pandas.util.testing import (assert_series_equal, assert_frame_equal,
                                 assert_numpy_array_equal)
from appelpy.diagnostics import (BadApples,
                                 partial_regression_plot, pp_plot, qq_plot,
                                 plot_residuals_vs_fitted_values,
                                 plot_residuals_vs_predictor_values)
from appelpy.linear_model import OLS
import matplotlib


def _reset_matplotlib():
    # Close open figures at start of new test
    plt.cla()
    plt.close('all')


@pytest.fixture(scope='module')
def df_prestige():
    df = sm.datasets.get_rdataset('Prestige', 'carData').data
    df['log2_income'] = np.log2(df['income'])
    return df


@pytest.fixture(scope='module')
def model_mtcars_short():
    df = sm.datasets.get_rdataset('mtcars').data
    X_list = ['cyl', 'wt', 'disp']
    model = OLS(df, ['mpg'], X_list).fit()
    return model


@pytest.mark.remote_data
def test_rvp_plot(model_mtcars_short):
    _reset_matplotlib()
    # Expected data, with x-vals sorted in asc order
    sorted_expected_data = (
        np.array([
            [1.513, 1.615, 1.835, 1.935, 2.14, 2.2, 2.32, 2.465, 2.62,
             2.77, 2.78, 2.875, 3.15, 3.17, 3.19, 3.215, 3.435, 3.44,
             3.44, 3.44, 3.46, 3.52, 3.57, 3.57, 3.73, 3.78, 3.84,
             3.845, 4.07, 5.25, 5.345, 5.424],
            [1.2222,  1.73801,  6.07224, -0.22323, -1.08655,  5.84247,
             -3.54021, -4.40346, -1.06821, -1.71077, -3.36495, -0.14111,
             -0.76771, -2.12603,  0.93363,  0.76267, -1.41135,  0.05625,
             1.68835, -1.34375, -1.39998, -0.90694, -2.23902, -1.09811,
             1.97191,  0.0537, -2.18265,  3.36188,  2.30804, -0.86805,
             4.01648, -0.14576]])).T

    with pytest.raises(ValueError):
        model_mtcars_short.diagnostic_plot('rvpplot', predictor='wt')
    with pytest.raises(ValueError):
        model_mtcars_short.diagnostic_plot('rvp_plot')
    with pytest.raises(ValueError):
        model_mtcars_short.diagnostic_plot('rvp_plot', predictor='weight')

    # ax=None
    _reset_matplotlib()
    fig = plot_residuals_vs_predictor_values(model_mtcars_short,
                                             predictor='wt',
                                             ax=None)
    assert isinstance(fig, matplotlib.figure.Figure)
    # Actual data:
    actual_data = fig.get_axes()[0].collections[0].get_offsets().data
    sorted_actual_data = actual_data[actual_data[:, 0].argsort()]
    assert_numpy_array_equal(np.round(sorted_actual_data, 5),
                             np.round(sorted_expected_data, 5))

    # Ax specified
    _reset_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    model_mtcars_short.diagnostic_plot('rvp_plot', predictor='wt', ax=ax)
    assert isinstance(fig, matplotlib.figure.Figure)

    # Actual data:
    actual_data = fig.get_axes()[0].collections[0].get_offsets().data
    sorted_actual_data = actual_data[actual_data[:, 0].argsort()]
    assert_numpy_array_equal(np.round(sorted_actual_data, 5),
                             sorted_expected_data)


@pytest.mark.remote_data
def test_rvf_plot(model_mtcars_short):
    _reset_matplotlib()

    # Expected data, with x-vals sorted in asc order
    sorted_expected_data = (
        np.array([[10.54576, 10.68352, 11.26805, 14.09196, 15.1463, 15.32809,
                   15.48265, 15.83812, 16.09811, 16.40694, 16.53902, 16.61135,
                   17.01165, 17.92603, 19.14375, 19.14375, 19.49998, 20.63733,
                   21.14111, 21.41077, 22.06821, 23.46637, 23.56771, 24.76495,
                   25.90346, 26.34021, 26.55753, 27.08655, 27.52323, 27.82776,
                   28.66199, 29.1778],
                  [-0.14576,  4.01648, -0.86805,  2.30804,  0.0537,  1.97191,
                   -2.18265,  3.36188, -1.09811, -0.90694, -2.23902, -1.41135,
                   1.68835, -2.12603, -1.34375,  0.05625, -1.39998,  0.76267,
                   -0.14111, -1.71077, -1.06821,  0.93363, -0.76771, -3.36495,
                   -4.40346, -3.54021,  5.84247, -1.08655, -0.22323,  6.07224,
                   1.73801,  1.2222]])).T

    with pytest.raises(ValueError):
        model_mtcars_short.diagnostic_plot('rvfplot')

    # Ax=None:
    _reset_matplotlib()
    fig = plot_residuals_vs_fitted_values(
        model_mtcars_short.results.resid,
        model_mtcars_short.results.fittedvalues, ax=None)
    assert isinstance(fig, matplotlib.figure.Figure)
    # Actual data:
    actual_data = fig.get_axes()[0].collections[0].get_offsets().data
    sorted_actual_data = actual_data[actual_data[:, 0].argsort()]
    assert_numpy_array_equal(np.round(sorted_actual_data, 5),
                             sorted_expected_data)

    # Ax specified
    _reset_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    model_mtcars_short.diagnostic_plot('rvf_plot', ax=ax)
    assert isinstance(fig, matplotlib.figure.Figure)
    # Actual data:
    actual_data = fig.get_axes()[0].collections[0].get_offsets().data
    sorted_actual_data = actual_data[actual_data[:, 0].argsort()]
    assert_numpy_array_equal(np.round(sorted_actual_data, 5),
                             sorted_expected_data)


@pytest.mark.remote_data
def test_lrv2_plot(model_mtcars_short):
    _reset_matplotlib()

    # Expected data
    leverage_vals = np.array([0.07142759, 0.07682215, 0.08269091, 0.05674927, 0.1408117,
                              0.04431891, 0.11526662, 0.15173483, 0.14598803, 0.14003649,
                              0.14003649, 0.17569423, 0.12832805, 0.13296335, 0.26025725,
                              0.26181188, 0.22799991, 0.08243088, 0.12760783, 0.09710846,
                              0.08749855, 0.08411163, 0.09265215, 0.07123245, 0.15365677,
                              0.0906371, 0.10659228, 0.17658798, 0.18857696, 0.09998757,
                              0.086823, 0.10155871])
    resid_vals = np.array([-0.42724911, -0.05660526, -1.42463347,  0.3026603,  0.70202004,
                           -0.55194795, -0.91745129,  0.39069709, -0.32018181,  0.02337847,
                           -0.55848373,  0.97978749,  0.8140355,  0.02222608, -0.38898636,
                           -0.0653876,  1.76184923,  2.35076364,  0.71718237,  2.46299294,
                           -1.77667992, -0.36524791, -0.57105687, -0.87289779,  1.40844822,
                           -0.09022259, -0.44305336,  0.5191179, -0.90965737, -0.6950216,
                           -0.44289589, -1.36824945])
    resid_sq_vals = np.square(resid_vals)

    expected_data = np.column_stack((resid_sq_vals, leverage_vals))
    sorted_expected_data = expected_data[expected_data[:, 0].argsort()]

    # Get actual data
    bad_apples = BadApples(model_mtcars_short).fit()
    assert_series_equal(bad_apples.y, model_mtcars_short.y)
    assert_frame_equal(bad_apples.X, model_mtcars_short.X)

    # 1a) default method call - ax=None
    _reset_matplotlib()
    fig = bad_apples.plot_leverage_vs_residuals_squared(ax=None)
    assert isinstance(fig, matplotlib.figure.Figure)
    actual_data = fig.get_axes()[0].collections[0].get_offsets().data
    sorted_actual_data = actual_data[actual_data[:, 0].argsort()]
    assert_numpy_array_equal(np.round(sorted_actual_data, 5),
                             np.round(sorted_expected_data, 5))

    # 1b) default method call - ax specified
    _reset_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    bad_apples.plot_leverage_vs_residuals_squared(ax=ax)
    assert isinstance(fig, matplotlib.figure.Figure)
    actual_data = fig.get_axes()[0].collections[0].get_offsets().data
    sorted_actual_data = actual_data[actual_data[:, 0].argsort()]
    assert_numpy_array_equal(np.round(sorted_actual_data, 5),
                             np.round(sorted_expected_data, 5))

    # 2) rescale=True
    _reset_matplotlib()
    fig = bad_apples.plot_leverage_vs_residuals_squared(rescale=True, ax=None)
    assert isinstance(fig, matplotlib.figure.Figure)

    obs = len(model_mtcars_short.X)

    expected_data = np.column_stack((np.divide(resid_sq_vals, obs),
                                    leverage_vals))
    sorted_expected_data = expected_data[expected_data[:, 0].argsort()]

    actual_data = fig.get_axes()[0].collections[0].get_offsets().data
    sorted_actual_data = actual_data[actual_data[:, 0].argsort()]

    assert_numpy_array_equal(np.round(sorted_actual_data, 5),
                             np.round(sorted_expected_data, 5))

    # 3) annotate=True
    _reset_matplotlib()
    fig = bad_apples.plot_leverage_vs_residuals_squared(annotate=True, ax=None)
    assert isinstance(fig, matplotlib.figure.Figure)
    expected_data = np.column_stack((resid_sq_vals, leverage_vals))
    sorted_expected_data = expected_data[expected_data[:, 0].argsort()]
    actual_data = fig.get_axes()[0].collections[0].get_offsets().data
    sorted_actual_data = actual_data[actual_data[:, 0].argsort()]

    annotations = [child.get_text() for child
                   in fig.get_axes()[0].get_children()
                   if isinstance(child, matplotlib.text.Annotation)]
    assert(len(annotations) == 20)


@pytest.mark.remote_data
def test_partial_regression_plot(df_prestige):
    _reset_matplotlib()
    model_wo_women = OLS(df_prestige,
                         ['prestige'],
                         ['education', 'log2_income'])
    model_w_women = OLS(df_prestige,
                        ['prestige'],
                        ['education', 'log2_income', 'women'])
    # Expected results with 'women' as added variable
    sorted_expected_data = (
        np.array([[-93.93836, -58.57753, -55.49617, -42.11369, -35.66954, -25.9112,
                   -25.86769, -24.58223, -23.7342, -23.54798, -23.01098, -22.44527,
                   -22.43929, -21.59109, -20.87776, -20.85103, -19.71815, -17.7076,
                   -16.90258, -16.7369, -15.39986, -15.28835, -15.19779, -14.53497,
                   -14.22256, -14.18416, -13.22735, -12.71817, -12.10991, -10.84503,
                   -10.18046,  -9.99679,  -9.95545,  -9.88944,  -9.56119,  -8.76271,
                   -7.64195,  -7.25597,  -6.94377,  -6.76293,  -6.63061,  -6.30956,
                   -6.29797,  -6.24731,  -5.49803,  -4.89141,  -4.81174,  -3.17664,
                   -3.13501,  -2.79909,  -2.56931,  -2.27721,  -1.70068,  -1.54629,
                   -0.9631,   0.64476,   0.91841,   1.22733,   1.22733,   1.61591,
                   1.65352,   2.25543,   2.26929,   3.91668,   4.34115,   4.52101,
                   6.80726,   6.91377,   7.02318,   8.64413,   8.71096,   8.97653,
                   9.26892,   9.65293,   9.68303,  12.12782,  14.08454,  14.42858,
                   14.65885,  14.84561,  17.89238,  18.70362,  19.17911,  21.10137,
                   23.81671,  26.76429,  30.25732,  30.57798,  30.82467,  31.08819,
                   31.66263,  32.14915,  32.79274,  33.10143,  34.62707,  36.7842,
                   37.204,  40.75253,  42.80259,  47.63141,  48.34697,  56.77472],
                  [-6.53339, -10.12183,  13.29246,  -2.49907,   8.29782,  -7.1847,
                   -3.00326,   3.73931,   6.83264,  -9.03795,  -1.05147,  -6.92754,
                   18.12704,   4.61055,   3.36723,   9.86312,  -2.49511,  -0.21752,
                   -13.28352,  -7.45875,  -9.73572, -11.53542,   2.18018,  -4.13201,
                   -0.89664,   4.15348,   4.25081,   1.77545,   2.697,  -6.07571,
                   -13.00251,  -0.94581,   8.32115,   0.43018,  -5.53771,   5.1915,
                   -3.05204,   3.39828,   8.14251,   8.00605,   3.61255,   3.21329,
                   -2.97429,  -6.90512,   2.32673,  -2.20231,   3.01192,  -3.2081,
                   5.55919,   2.12583,   4.92325,  -3.35745,  -1.77673,   1.54191,
                   -8.64127,   2.97265,   3.77047,  -7.89845,   1.70155,  -3.33801,
                   -12.28337,   6.17828,  -7.01689,  -9.20361,  -4.71027,   9.3336,
                   -1.89465,  -0.57515, -17.03456,  -1.99676,  -8.43499,  10.71922,
                   8.50111, -14.83202,  -6.6973,   3.73437,   1.91247,   2.55986,
                   5.65674,   2.41152,  -8.3028, -12.12295,   4.21888,  -3.25473,
                   3.47491,   5.66643,  -0.9881,  13.68947,  -3.0455,   1.66228,
                   15.16382,  -4.91281,  -1.47625,  -1.01092,   2.46563,  -1.01171,
                   1.47855,  16.2411,  -0.93693,  13.53354,  -0.15396,   6.88775]])
    ).T

    # model_wo_women - ax=None
    fig = partial_regression_plot(model_wo_women, df_prestige, 'women', ax=None)
    assert isinstance(fig, matplotlib.figure.Figure)
    actual_data = fig.get_axes()[0].collections[0].get_offsets().data
    sorted_actual_data = actual_data[actual_data[:, 0].argsort()]
    assert_numpy_array_equal(np.round(sorted_actual_data, 5),
                             sorted_expected_data)

    # model_wo_women - ax specified
    _reset_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    partial_regression_plot(model_wo_women, df_prestige, 'women', ax=ax)
    assert isinstance(fig, matplotlib.figure.Figure)
    actual_data = fig.get_axes()[0].collections[0].get_offsets().data
    sorted_actual_data = actual_data[actual_data[:, 0].argsort()]
    assert_numpy_array_equal(np.round(sorted_actual_data, 5),
                             sorted_expected_data)

    # model_w_women
    _reset_matplotlib()
    fig = partial_regression_plot(model_w_women, df_prestige, 'women', ax=None)
    assert isinstance(fig, matplotlib.figure.Figure)
    actual_data = fig.get_axes()[0].collections[0].get_offsets().data
    sorted_actual_data = actual_data[actual_data[:, 0].argsort()]
    assert_numpy_array_equal(np.round(sorted_actual_data, 5),
                             sorted_expected_data)

    # Error: added variable not in df:
    with pytest.raises(ValueError):
        partial_regression_plot(model_wo_women, df_prestige, 'men')

    # Error: NaN value in added variable col:
    with pytest.raises(ValueError):
        error_df = df_prestige.copy()
        error_df.loc[0, 'women'] = np.NaN
        partial_regression_plot(model_wo_women, error_df, 'women')


@pytest.mark.remote_data
def test_qq_plot(model_mtcars_short):
    _reset_matplotlib()

    # ax=None
    fig = qq_plot(model_mtcars_short.resid, ax=None)
    assert isinstance(fig, matplotlib.figure.Figure)
    ax = fig.get_axes()[0]
    assert(ax.get_xlabel() == 'Theoretical Quantiles')
    assert(ax.get_ylabel() == 'Sample Quantiles')

    # Ax specified
    _reset_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    model_mtcars_short.diagnostic_plot('qq_plot', ax=ax)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert(ax.get_xlabel() == 'Theoretical Quantiles')
    assert(ax.get_ylabel() == 'Sample Quantiles')


@pytest.mark.remote_data
def test_pp_plot(model_mtcars_short):
    _reset_matplotlib()

    # ax=None
    fig = pp_plot(model_mtcars_short.resid, ax=None)
    assert isinstance(fig, matplotlib.figure.Figure)
    ax = fig.get_axes()[0]
    assert(ax.get_xlabel() == 'Theoretical Probabilities')
    assert(ax.get_ylabel() == 'Sample Probabilities')

    # Ax specified
    _reset_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    model_mtcars_short.diagnostic_plot('pp_plot', ax=ax)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert(ax.get_xlabel() == 'Theoretical Probabilities')
    assert(ax.get_ylabel() == 'Sample Probabilities')
