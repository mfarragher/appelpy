import pytest
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pandas.util.testing import (assert_series_equal,
                                 assert_numpy_array_equal)
from appelpy.diagnostics import (variance_inflation_factors, BadApples,
                                 heteroskedasticity_test, wald_test)
from appelpy.linear_model import OLS
from appelpy.discrete_model import Logit


def _round_significant_figures(scalar, n_figures):
    return (round(scalar,
                  (-int(np.floor(np.sign(scalar) * np.log10(abs(scalar))))
                   + n_figures - 1)))


@pytest.fixture(scope='module')
def model_mtcars_long():
    df = sm.datasets.get_rdataset('mtcars').data
    X_list = ['cyl', 'disp', 'hp', 'drat', 'wt',
              'qsec', 'vs', 'am', 'gear', 'carb']
    model = OLS(df, ['mpg'], X_list).fit()
    return model


@pytest.fixture(scope='module')
def model_journals():
    df = sm.datasets.get_rdataset('Journals', 'Ecdat').data

    # Model
    df['ln_citestot'] = np.log(df['citestot'])
    df['ln_oclc'] = np.log(df['oclc'])
    df['ln_ppc'] = np.log(df['libprice'] / df['citestot'])

    model = OLS(df, ['ln_oclc'], ['ln_ppc']).fit()
    return model


@pytest.fixture(scope='module')
def model_cars():
    df = sm.datasets.get_rdataset('cars').data
    X_list = ['speed']
    model = OLS(df, ['dist'], X_list).fit()
    return model


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
def model_caschools():
    df = sm.datasets.get_rdataset('Caschool', 'Ecdat').data

    # Feature
    df['expnstu_1000'] = df['expnstu'] / 1000
    # Model
    X_list = ['str', 'elpct', 'expnstu_1000']
    model_hc1 = OLS(df, ['testscr'], X_list).fit()
    return model_hc1


@pytest.fixture(scope='module')
def model_longley():
    df = sm.datasets.get_rdataset('longley').data

    # Pre-processing - column names:
    df.columns = (df.columns
                  .str.replace(r"[ ,.,-]", '_')
                  .str.lower())
    # Pre-processing - match the longley from statsmodels.datasets:
    for col in ['gnp', 'population', 'employed']:
        df[col] = df[col] * 1000
    for col in ['armed_forces', 'unemployed']:
        df[col] = df[col] * 10

    # Model
    X_list = ['gnp_deflator', 'gnp', 'unemployed', 'armed_forces',
              'population', 'year']
    model = OLS(df, ['employed'], X_list).fit()
    return model


@pytest.fixture(scope='module')
def bad_apples_hills():
    df = sm.datasets.get_rdataset('hills', 'MASS').data

    model = OLS(df, ['time'], ['dist', 'climb']).fit()

    bad_apples = BadApples(model).fit()

    return bad_apples


@pytest.mark.remote_data
def test_model_not_fitted():
    df = sm.datasets.get_rdataset('cars').data
    X_list = ['speed']
    model = OLS(df, ['dist'], X_list)

    assert not model.is_fitted


@pytest.mark.remote_data
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


@pytest.mark.remote_data
def test_bad_apple_prints(capsys):
    df = sm.datasets.get_rdataset('hills', 'MASS').data

    model = OLS(df, ['time'], ['dist', 'climb']).fit()

    BadApples(model).fit(printing=True)

    captured = capsys.readouterr()
    expected_print = "Calculating influence measures...\nCalculations saved to object.\n"
    assert captured.out == expected_print


@pytest.mark.remote_data
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


@pytest.mark.remote_data
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


@pytest.mark.remote_data
def test_outliers(bad_apples_hills):
    assert (bad_apples_hills.measures_outliers
            ['resid_studentized'].idxmax() == 'Knock Hill')
    assert (bad_apples_hills.measures_outliers
            ['resid_standardized'].idxmax() == 'Knock Hill')

    expected_max_resid_student = 7.610845
    assert(np.round(bad_apples_hills.measures_outliers
                    ['resid_studentized'].max(), 6)
           == expected_max_resid_student)

    # Measurement error - expected outlier
    assert (set({'Knock Hill'})
            .issubset(
                set(bad_apples_hills.indices_outliers)))


@pytest.mark.remote_data
def test_heteroskedasticity_diagnostics(model_cars, model_journals):
    # Check for invalid arguments first:
    with pytest.raises(ValueError):
        heteroskedasticity_test(model_cars, 'bp')
        heteroskedasticity_test(model_cars, 100)

    expected_keys = set(['distribution', 'nu', 'test_stat', 'p_value'])

    # Test on actual data
    actual_bp_results = heteroskedasticity_test('breusch_pagan',
                                                model_cars)
    expected_bp_results = {'distribution': 'chi2', 'nu': 1,
                           'test_stat': 4.650233, 'p_value': 0.03104933}
    assert (np.round(actual_bp_results['test_stat'], 6)
            == expected_bp_results['test_stat'])
    assert (np.round(actual_bp_results['p_value'], 8)
            == expected_bp_results['p_value'])
    for k in ['distribution', 'nu']:
        assert actual_bp_results[k] == expected_bp_results[k]
    assert set(actual_bp_results.keys()) == expected_keys

    actual_bp_results = heteroskedasticity_test('breusch_pagan_studentized',
                                                model_journals)
    expected_bp_results = {'distribution': 'chi2', 'nu': 1,
                           'test_stat': 9.8, 'p_value': 0.002}
    assert (np.round(actual_bp_results['test_stat'], 1)
            == expected_bp_results['test_stat'])
    assert (np.round(actual_bp_results['p_value'], 3)
            == expected_bp_results['p_value'])
    for k in ['distribution', 'nu']:
        assert actual_bp_results[k] == expected_bp_results[k]
    assert set(actual_bp_results.keys()) == expected_keys

    actual_bp_results = heteroskedasticity_test('breusch_pagan_studentized',
                                                model_cars)
    expected_bp_results = {'distribution': 'chi2', 'nu': 1,
                           'test_stat': 3.2149, 'p_value': 0.07297}
    assert (np.round(actual_bp_results['test_stat'], 4)
            == expected_bp_results['test_stat'])
    assert (np.round(actual_bp_results['p_value'], 5)
            == expected_bp_results['p_value'])
    for k in ['distribution', 'nu']:
        assert actual_bp_results[k] == expected_bp_results[k]
    assert set(actual_bp_results.keys()) == expected_keys

    expected_w_results = {'distribution': 'chi2', 'nu': 2,
                          'test_stat': 11, 'p_value': 0.004}
    actual_w_results = heteroskedasticity_test('white', model_journals)
    assert (np.round(actual_w_results['test_stat'], 0)
            == expected_w_results['test_stat'])
    assert (np.round(actual_w_results['p_value'], 3)
            == expected_w_results['p_value'])
    for k in ['distribution', 'nu']:
        assert actual_w_results[k] == expected_w_results[k]
    assert set(actual_w_results.keys()) == expected_keys

    # Regressors subset (check that passed list works)
    with pytest.raises(ValueError):
        heteroskedasticity_test('breusch_pagan_studentized',
                                model_cars,
                                regressors_subset=['time'])
    with pytest.raises(ValueError):
        heteroskedasticity_test('breusch_pagan',
                                model_cars,
                                regressors_subset=['time'])

    actual_bp_results = heteroskedasticity_test('breusch_pagan',
                                                model_cars,
                                                regressors_subset=['speed'])
    expected_bp_results = {'distribution': 'chi2', 'nu': 1,
                           'test_stat': 4.650233, 'p_value': 0.03104933}
    assert (np.round(actual_bp_results['test_stat'], 6)
            == expected_bp_results['test_stat'])
    assert (np.round(actual_bp_results['p_value'], 8)
            == expected_bp_results['p_value'])
    for k in ['distribution', 'nu']:
        assert actual_bp_results[k] == expected_bp_results[k]
    assert set(actual_bp_results.keys()) == expected_keys

    actual_bp_results = heteroskedasticity_test('breusch_pagan_studentized',
                                                model_cars,
                                                regressors_subset=['speed'])
    expected_bp_results = {'distribution': 'chi2', 'nu': 1,
                           'test_stat': 3.2149, 'p_value': 0.07297}
    assert (np.round(actual_bp_results['test_stat'], 4)
            == expected_bp_results['test_stat'])
    assert (np.round(actual_bp_results['p_value'], 5)
            == expected_bp_results['p_value'])
    for k in ['distribution', 'nu']:
        assert actual_bp_results[k] == expected_bp_results[k]
    assert set(actual_bp_results.keys()) == expected_keys

    expected_w_results = {'distribution': 'chi2', 'nu': 2,
                          'test_stat': 11, 'p_value': 0.004}
    actual_w_results = heteroskedasticity_test('white', model_journals,
                                               regressors_subset=['ln_ppc'])
    assert (np.round(actual_w_results['test_stat'], 0)
            == expected_w_results['test_stat'])
    assert (np.round(actual_w_results['p_value'], 3)
            == expected_w_results['p_value'])
    for k in ['distribution', 'nu']:
        assert actual_w_results[k] == expected_w_results[k]
    assert set(actual_w_results.keys()) == expected_keys


@pytest.mark.remote_data
def test_wald_test_output(model_wells, model_caschools, model_longley):
    expected_dict_keys = set(['distribution', 'nu', 'test_stat', 'p_value'])

    # 1) Logit
    # model_wells Wald test in R:
    #
    # library(carData)
    # library(survey)
    #
    # data(Wells)
    #
    # model <- glm(formula = switch ~ arsenic + distance + education + association,
    # data=Wells, family=binomial())
    #
    # regTermTest(model, ~association+education, df=Inf)

    actual_wells_output = wald_test(model_wells,
                                    ['association', 'education'])
    expected_wells_output_rounded = {'distribution': 'chi2', 'nu': 2,
                                     'test_stat': 22.53277,
                                     'p_value': 1.2796e-05}
    assert (actual_wells_output['distribution']
            == expected_wells_output_rounded['distribution'])
    assert (np.round(actual_wells_output['test_stat'], 5)
            == expected_wells_output_rounded['test_stat'])
    assert (_round_significant_figures(actual_wells_output['p_value'], 5)
            == expected_wells_output_rounded['p_value'])
    assert set(actual_wells_output.keys()) == expected_dict_keys

    # 2) OLS - caschools
    actual_caschools_output = wald_test(model_caschools,
                                        {'str': 0, 'expnstu_1000': 0})
    expected_caschools_output_rounded = {'distribution': 'F', 'nu': 2,
                                         'test_stat': 8.0101,
                                         'p_value': 0.000386}
    assert (actual_caschools_output['distribution']
            == expected_caschools_output_rounded['distribution'])
    assert (np.round(actual_caschools_output['test_stat'], 4)
            == expected_caschools_output_rounded['test_stat'])
    assert (_round_significant_figures(actual_caschools_output['p_value'], 3)
            == expected_caschools_output_rounded['p_value'])
    assert set(actual_caschools_output.keys()) == expected_dict_keys

    # 3) OLS - Longley
    actual_longley_output = wald_test(model_longley,
                                      {('gnp_deflator', 'gnp'): 0,
                                       'unemployed': 2,
                                       'year': 1829})
    expected_longley_output_rounded = {'distribution': 'F', 'nu': 3,
                                       'test_stat': 144.1798,
                                       'p_value': 6.322e-08}
    assert (actual_longley_output['distribution']
            == expected_longley_output_rounded['distribution'])
    assert (np.round(actual_longley_output['test_stat'], 4)
            == expected_longley_output_rounded['test_stat'])
    assert (_round_significant_figures(actual_longley_output['p_value'], 4)
            == expected_longley_output_rounded['p_value'])
    assert set(actual_longley_output.keys()) == expected_dict_keys

    # Invalid arguments:
    with pytest.raises(ValueError):
        wald_test(model_caschools, ['INVALID', 'str'])
    with pytest.raises(TypeError):
        wald_test(model_caschools, {'str': 'non-scalar', 'expnstu_1000': 0})
    with pytest.raises(ValueError):
        wald_test(model_caschools, {'INVALID': 0, 'str': 0})
    with pytest.raises(ValueError):
        wald_test(model_caschools, {('str', 'expnstu_1000', 'elpct'): 0})
    with pytest.raises(ValueError):
        wald_test(model_caschools, {('str', 'INVALID'): 0})
    with pytest.raises(TypeError):
        wald_test(model_caschools, {0: 0})
    with pytest.raises(TypeError):
        wald_test(model_caschools, 'bad_argument')
