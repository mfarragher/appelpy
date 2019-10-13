import pytest
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pandas.util.testing import (assert_series_equal, assert_frame_equal,
                                 assert_numpy_array_equal)
from appelpy.diagnostics import (variance_inflation_factors, BadApples,
                                 heteroskedasticity_test,
                                 partial_regression_plot)
from appelpy.linear_model import OLS


@pytest.fixture(scope='module')
def df_prestige():
    df = sm.datasets.get_rdataset('Prestige', 'carData').data
    df['log2_income'] = np.log2(df['income'])
    return df


@pytest.fixture(scope='module')
def model_mtcars_long():
    df = sm.datasets.get_rdataset('mtcars').data
    X_list = ['cyl', 'disp', 'hp', 'drat', 'wt',
              'qsec', 'vs', 'am', 'gear', 'carb']
    model = OLS(df, ['mpg'], X_list)
    return model


@pytest.fixture(scope='module')
def model_mtcars_short():
    df = sm.datasets.get_rdataset('mtcars').data
    X_list = ['cyl', 'wt', 'disp']
    model = OLS(df, ['mpg'], X_list)
    return model


@pytest.fixture(scope='module')
def model_journals():
    df = sm.datasets.get_rdataset('Journals', 'Ecdat').data

    # Model
    df['ln_citestot'] = np.log(df['citestot'])
    df['ln_oclc'] = np.log(df['oclc'])
    df['ln_ppc'] = np.log(df['libprice'] / df['citestot'])

    model = OLS(df, ['ln_oclc'], ['ln_ppc'])
    return model


@pytest.fixture(scope='module')
def model_cars():
    df = sm.datasets.get_rdataset('cars').data
    X_list = ['speed']
    model = OLS(df, ['dist'], X_list)
    return model


@pytest.fixture(scope='module')
def bad_apples_hills():
    df = sm.datasets.get_rdataset('hills', 'MASS').data

    model = OLS(df, ['time'], ['dist', 'climb'])

    bad_apples = BadApples(model)

    return bad_apples


def test_variance_inflation_factors(model_mtcars_long):
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
    df_vif = variance_inflation_factors(model_mtcars_long.X)
    assert_series_equal(df_vif['VIF'].round(6), expected_vif)


def test_influence(bad_apples_hills):
    indices = ['Greenmantle', 'Carnethy', 'Craig Dunain', 'Ben Rha', 'Ben Lomond',
               'Goatfell', 'Bens of Jura', 'Cairnpapple', 'Scolty', 'Traprain',
               'Lairig Ghru', 'Dollar', 'Lomonds', 'Cairn Table', 'Eildon Two',
               'Cairngorm', 'Seven Hills', 'Knock Hill', 'Black Hill', 'Creag Beag',
               'Kildcon Hill', 'Meall Ant-Suidhe', 'Half Ben Nevis', 'Cow Hill',
               'N Berwick Law', 'Creag Dubh', 'Burnswark', 'Largo Law', 'Criffel',
               'Acmony', 'Ben Nevis', 'Knockfarrel', 'Two Breweries', 'Cockleroi',
               'Moffat Chase']

    expected_dfbeta_const = (pd.Series(
        np.array([0.03781, -0.05958, -0.04858, -0.00766, -0.05046, 0.00348,
                  -0.89065, -0.00844, -0.01437, 0.04703, -0.30118, -0.01149,
                  -0.03173, 0.11803, -0.10038, -0.01852, 0.01196, 1.75827,
                  -0.15889, 0.00866, 0.04777, -0.01889, -0.04131, 0.07483,
                  0.03691, -0.13772, -0.0292, -0.04764, -0.00214, -0.08532,
                  0.02099, -0.02858, -0.15823, -0.00356, 0.20872]),
        index=indices, name='dfbeta_const'))
    assert_series_equal(np.round(bad_apples_hills
                                 .measures_influence['dfbeta_const'], 5),
                        expected_dfbeta_const)

    expected_dfbeta_dist = (pd.Series(
        np.array([-1.66140e-02, 6.72150e-02, -6.70700e-03, -5.67500e-03,
                  8.47090e-02, -4.31600e-03, -7.12774e-01, -1.64800e-03,
                  9.13000e-04, 1.30570e-02, 7.68716e-01, 9.65600e-03,
                  -2.99110e-02, 4.20340e-02, 5.77010e-02, 6.78900e-03,
                  -6.65050e-02, -4.06545e-01, 4.43110e-02, 1.42400e-03,
                  -1.00190e-02, 1.38560e-02, 3.40970e-02, -4.63850e-02,
                  -1.26330e-02, 1.36124e-01, -5.70200e-03, 6.93600e-03,
                  6.47000e-04, -7.70500e-03, 1.70124e-01, -8.69400e-03,
                  9.70140e-02, 7.04000e-04, -1.99048e-01]),
        index=indices, name='dfbeta_dist'))
    assert_series_equal(np.round(bad_apples_hills
                                 .measures_influence['dfbeta_dist'], 6),
                        expected_dfbeta_dist)

    expected_dfbeta_climb = (pd.Series(
        np.array([-4.744000e-03, -7.339600e-02, 2.803300e-02, 8.764000e-03,
                  -1.450050e-01, 7.576000e-03, 2.364618e+00, 5.562000e-03,
                  6.161000e-03, -3.651900e-02, -4.798490e-01, -7.488000e-03,
                  -7.070000e-04, -1.048840e-01, -2.231700e-02, -9.986200e-02,
                  3.445500e-02, -6.559340e-01, 2.941400e-02, -5.946000e-03,
                  -1.919900e-02, -6.465000e-03, -3.302200e-02, 6.428000e-03,
                  -8.257000e-03, -1.013060e-01, 1.923900e-02, 1.499000e-02,
                  -3.280000e-04, 5.483800e-02, -3.736340e-01, 2.327500e-02,
                  1.557020e-01, 1.054000e-03, -1.009070e-01]),
        index=indices, name='dfbeta_climb'))
    assert_series_equal(np.round(bad_apples_hills
                                 .measures_influence['dfbeta_climb'], 6),
                        expected_dfbeta_climb)

    expected_dffits = (pd.Series(
        np.array([0.03862, -0.11956, -0.0631, -0.01367, -0.20947, 0.01221,
                  2.69909, -0.01115, -0.01663, 0.06399, 0.78569, -0.01672,
                  -0.1177, 0.1661, -0.1192, -0.21135, -0.08337, 1.84237,
                  -0.17484, 0.01102, 0.05032, -0.02234, -0.06961, 0.07839,
                  0.03808, -0.19782, -0.03857, -0.05446, -0.00309, -0.10362,
                  -0.44138, -0.03931, 0.33384, -0.00392, -0.39445]),
        index=indices, name='dffits'))
    assert_series_equal(np.round(bad_apples_hills
                                 .measures_influence['dffits'], 5),
                        expected_dffits)

    expected_cooks_d = (
        np.array([5.13e-04, 4.88e-03, 1.37e-03, 6.43e-05, 1.47e-02, 5.13e-05,
                  1.89e+00, 4.28e-05, 9.52e-05, 1.41e-03, 2.11e-01, 9.61e-05,
                  4.70e-03, 9.34e-03, 4.83e-03, 1.49e-02, 2.39e-03, 4.07e-01,
                  1.03e-02, 4.18e-05, 8.70e-04, 1.72e-04, 1.66e-03, 2.11e-03,
                  4.99e-04, 1.32e-02, 5.11e-04, 1.02e-03, 3.29e-06, 3.67e-03,
                  6.41e-02, 5.31e-04, 3.77e-02, 5.29e-06, 5.24e-02]))
    actual_cooks_d = (bad_apples_hills
                      .measures_influence['cooks_d'].to_numpy())
    precision_formatter = np.vectorize(lambda f: format(f, '6.2E'))
    actual_cooks_d = precision_formatter(actual_cooks_d)
    expected_cooks_d = precision_formatter(expected_cooks_d)
    assert_numpy_array_equal(actual_cooks_d,
                             expected_cooks_d)

    assert (set({'Bens of Jura', 'Lairig Ghru', 'Knock Hill'})
            .issubset(
                set(bad_apples_hills.indices_high_influence)))

    extremes_df = bad_apples_hills.show_extreme_observations()
    assert (set({'Bens of Jura', 'Lairig Ghru', 'Knock Hill'})
            .issubset(
                set(extremes_df.index.tolist())))


def test_leverage(bad_apples_hills):
    indices = ['Greenmantle', 'Carnethy', 'Craig Dunain', 'Ben Rha', 'Ben Lomond',
               'Goatfell', 'Bens of Jura', 'Cairnpapple', 'Scolty', 'Traprain',
               'Lairig Ghru', 'Dollar', 'Lomonds', 'Cairn Table', 'Eildon Two',
               'Cairngorm', 'Seven Hills', 'Knock Hill', 'Black Hill', 'Creag Beag',
               'Kildcon Hill', 'Meall Ant-Suidhe', 'Half Ben Nevis', 'Cow Hill',
               'N Berwick Law', 'Creag Dubh', 'Burnswark', 'Largo Law', 'Criffel',
               'Acmony', 'Ben Nevis', 'Knockfarrel', 'Two Breweries', 'Cockleroi',
               'Moffat Chase']

    expected_leverage = pd.Series(
        np.array([0.0538, 0.0495, 0.0384, 0.0485, 0.0553, 0.0468, 0.4204, 0.041,
                  0.0403, 0.0457, 0.6898, 0.0435, 0.0323, 0.0513, 0.0388, 0.0444,
                  0.0831, 0.0554, 0.0385, 0.0459, 0.0566, 0.0483, 0.0398, 0.0584,
                  0.0507, 0.055, 0.041, 0.0376, 0.0299, 0.0482, 0.1216, 0.0475,
                  0.1716, 0.0403, 0.191]),
        index=indices, name='leverage')

    assert_series_equal(np.round(bad_apples_hills
                                 .measures_leverage, 4),
                        expected_leverage)

    # Highest race and longest race - expected high leverage
    assert (set({'Bens of Jura', 'Lairig Ghru'})
            .issubset(
                set(bad_apples_hills.indices_high_leverage)))


def test_outliers(bad_apples_hills):
    assert (bad_apples_hills.measures_outliers
            ['resid_studentized'].idxmax() == 'Knock Hill')
    assert (bad_apples_hills.measures_outliers
            ['resid_standardized'].idxmax() == 'Knock Hill')

    expected_max_resid_student = 7.610845
    assert(np.round(bad_apples_hills.measures_outliers
                    ['resid_studentized'].max(), 6) == expected_max_resid_student)

    # Measurement error - expected outlier
    assert (set({'Knock Hill'})
            .issubset(
                set(bad_apples_hills.indices_outliers)))


def test_heteroskedasticity_diagnostics(model_cars, model_journals):
    # Check for invalid arguments first:
    with pytest.raises(ValueError):
        heteroskedasticity_test(model_cars, 'bp')
        heteroskedasticity_test(model_cars, 100)

    # Test on actual data
    expected_lm, expected_pval = (4.650233, 0.03104933)
    lm, pval = heteroskedasticity_test('breusch_pagan',
                                       model_cars)
    assert (np.round(lm, 6) == expected_lm)
    assert (np.round(pval, 8) == expected_pval)

    expected_lm, expected_pval = (9.8, 0.002)
    lm, pval = heteroskedasticity_test('breusch_pagan_studentized',
                                       model_journals)
    assert (np.round(lm, 1) == expected_lm)
    assert (np.round(pval, 3) == expected_pval)

    expected_lm, expected_pval = (3.2149, 0.07297)
    lm, pval = heteroskedasticity_test('breusch_pagan_studentized',
                                       model_cars)
    assert (np.round(lm, 4) == expected_lm)
    assert (np.round(pval, 5) == expected_pval)

    expected_lm, expected_pval = (11, 0.004)
    lm, pval = heteroskedasticity_test('white', model_journals)
    assert (np.round(lm, 0) == expected_lm)
    assert (np.round(pval, 3) == expected_pval)

    # Regressors subset (check that passed list works)
    with pytest.raises(ValueError):
        heteroskedasticity_test('breusch_pagan_studentized',
                                model_cars,
                                regressors_subset=['time'])
    with pytest.raises(ValueError):
        heteroskedasticity_test('breusch_pagan',
                                model_cars,
                                regressors_subset=['time'])

    expected_lm, expected_pval = (4.650233, 0.03104933)
    lm, pval = heteroskedasticity_test('breusch_pagan',
                                       model_cars,
                                       regressors_subset=['speed'])
    assert (np.round(lm, 6) == expected_lm)
    assert (np.round(pval, 8) == expected_pval)

    expected_lm, expected_pval = (3.2149, 0.07297)
    lm, pval = heteroskedasticity_test('breusch_pagan_studentized',
                                       model_cars,
                                       regressors_subset=['speed'])
    assert (np.round(lm, 4) == expected_lm)
    assert (np.round(pval, 5) == expected_pval)

    expected_lm, expected_pval = (11, 0.004)
    lm, pval = heteroskedasticity_test('white', model_journals,
                                       regressors_subset=['ln_ppc'])
    assert (np.round(lm, 0) == expected_lm)
    assert (np.round(pval, 3) == expected_pval)


def test_rvf_plot(model_mtcars_short):
    # Close open figures at start of new test
    plt.cla()
    plt.close('all')

    fig = model_mtcars_short.diagnostic_plot('rvf_plot')
    assert isinstance(fig, matplotlib.figure.Figure)

    # Expected data, with x-vals sorted in asc order
    sorted_expected_data = (
        np.array([[10.4, 10.4, 13.3, 14.3, 14.7, 15.,
                   15.2, 15.2, 15.5, 15.8, 16.4, 17.3,
                   17.8, 18.1, 18.7, 19.2, 19.2, 19.7,
                   21., 21., 21.4, 21.4, 21.5, 22.8,
                   22.8, 24.4, 26., 27.3, 30.4, 30.4,
                   32.4, 33.9],
                  [-0.14576, -0.86805, -2.18265, -2.23902,  4.01648, -1.09811,
                   0.0537, -1.41135, -0.90694, -2.12603,  2.30804,  1.97191,
                   -1.34375, -1.39998,  1.68835,  0.05625,  3.36188, -1.71077,
                   -1.06821, -0.14111,  0.76267, -3.36495, -4.40346, -0.76771,
                   -3.54021,  0.93363, -1.08655, -0.22323,  1.2222,  1.73801,
                   5.84247,  6.07224]])).T

    # Actual data:
    actual_data = fig.get_axes()[0].collections[0].get_offsets().data
    sorted_actual_data = actual_data[actual_data[:, 0].argsort()]

    assert_numpy_array_equal(np.round(sorted_actual_data, 5),
                             sorted_expected_data)


def test_rvp_plot(model_mtcars_short):
    # Close open figures at start of new test
    plt.cla()
    plt.close('all')

    with pytest.raises(ValueError):
        model_mtcars_short.diagnostic_plot('rvpplot')

    plt.cla()
    plt.close('all')
    fig = model_mtcars_short.diagnostic_plot('rvp_plot')
    assert isinstance(fig, matplotlib.figure.Figure)

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

    # Actual data:
    actual_data = fig.get_axes()[0].collections[0].get_offsets().data
    sorted_actual_data = actual_data[actual_data[:, 0].argsort()]

    assert_numpy_array_equal(np.round(sorted_actual_data, 5),
                             sorted_expected_data)


def test_lrv2_plot(model_mtcars_short):
    # Close open figures at start of new test
    plt.cla()
    plt.close('all')

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
    bad_apples = BadApples(model_mtcars_short)
    assert_series_equal(bad_apples.y, model_mtcars_short.y)
    assert_frame_equal(bad_apples.X, model_mtcars_short.X)
    # 1) default method call
    fig = bad_apples.plot_leverage_vs_residuals_squared()
    assert isinstance(fig, matplotlib.figure.Figure)

    actual_data = fig.get_axes()[0].collections[0].get_offsets().data
    sorted_actual_data = actual_data[actual_data[:, 0].argsort()]

    assert_numpy_array_equal(np.round(sorted_actual_data, 5),
                             np.round(sorted_expected_data, 5))

    # 2) rescale=True
    plt.cla()
    plt.close('all')
    fig = bad_apples.plot_leverage_vs_residuals_squared(rescale=True)
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
    plt.cla()
    plt.close('all')
    fig = bad_apples.plot_leverage_vs_residuals_squared(annotate=True)
    assert isinstance(fig, matplotlib.figure.Figure)

    expected_data = np.column_stack((resid_sq_vals, leverage_vals))
    sorted_expected_data = expected_data[expected_data[:, 0].argsort()]

    actual_data = fig.get_axes()[0].collections[0].get_offsets().data
    sorted_actual_data = actual_data[actual_data[:, 0].argsort()]

    annotations = [child.get_text() for child
                   in fig.get_axes()[0].get_children()
                   if isinstance(child, matplotlib.text.Annotation)]
    assert(len(annotations) == 20)


def test_partial_regression_plot(df_prestige):
    # Close open figures at start of new test
    plt.cla()
    plt.close('all')

    model_wo_women = OLS(df_prestige,
                         ['prestige'],
                         ['education', 'log2_income'])

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

    # model_wo_women
    fig = partial_regression_plot(model_wo_women, df_prestige, 'women')
    assert isinstance(fig, matplotlib.figure.Figure)
    actual_data = fig.get_axes()[0].collections[0].get_offsets().data
    sorted_actual_data = actual_data[actual_data[:, 0].argsort()]
    assert_numpy_array_equal(np.round(sorted_actual_data, 5),
                             sorted_expected_data)

    # Error: added variable not in df:
    plt.cla()
    plt.close('all')
    with pytest.raises(ValueError):
        partial_regression_plot(model_wo_women, df_prestige, 'men')

    # Error: NaN value in added variable col:
    plt.cla()
    plt.close('all')
    with pytest.raises(ValueError):
        error_df = df_prestige.copy()
        error_df.loc[0, 'women'] = np.NaN
        partial_regression_plot(model_wo_women, error_df, 'women')


def test_qq_plot(model_mtcars_short):
    # Close open figures at start of new test
    plt.cla()
    plt.close('all')
    fig = model_mtcars_short.diagnostic_plot('qq_plot')
    assert isinstance(fig, matplotlib.figure.Figure)

    ax = model_mtcars_short.diagnostic_plot('qq_plot').get_axes()[0]
    assert(ax.get_xlabel() == 'Theoretical Quantiles')
    assert(ax.get_ylabel() == 'Sample Quantiles')


def test_pp_plot(model_mtcars_short):
    # Close open figures at start of new test
    plt.cla()
    plt.close('all')
    fig = model_mtcars_short.diagnostic_plot('pp_plot')
    assert isinstance(fig, matplotlib.figure.Figure)

    ax = model_mtcars_short.diagnostic_plot('pp_plot').get_axes()[0]
    assert(ax.get_xlabel() == 'Theoretical Probabilities')
    assert(ax.get_ylabel() == 'Sample Probabilities')
