<header>
<pre><p style="font-size:28px;"><b>discrete_model</b></p></pre>
</header>

# Overview
These are the classes for discrete choice models:

- Logistic regression (`Logit`)

The classes are built upon Statsmodels.

# Fit a model
[![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/mfarragher/appelpy-examples/blob/master/02-01_logistic-regression_glm-logit.ipynb): static render of the notebook that fits a Logit regression.  Predictions are also made using the original data to show the estimated probabilities of the positive class.

```python
from appelpy.discrete_model import Logit
model1 = Logit(df, y_list, X_list).fit()
model1.results_output  # returns summary results
```

The **`fit` method must be called** in order to set attributes for the model object.

There are three **important parameters** for initialising any model class in Appelpy:

- `df`: the dataframe to use for modelling.  This must have no NaN values, no infinite values, etc.
- `y_list`: a list with the dependent variable column name.
- `regressors_list`: a list of column names for independent variables (regressors).

## Attributes
Here are some attributes available for discrete choice models:

- `y` and `X`: the dataframes of the dependent and independent variables.
- `y_standardized` and `X_standardized`: the standardized versions of `y` and `X`.
- `results_output` for the Statsmodels summary of the model.  Note: the Statsmodels results object is also stored in the `results` attribute.
- `results_output_standardized` for the standardized estimates of the model.
- `model_selection_stats`: dictionary of key statistics on the model fit.
- The model residuals `resid` and their standardized form `resid_standardized`.

Logit has the `odds_ratio` attribute.

## Methods
For all model classes there is a `significant_regressors` method that returns a list of the significant independent variables of a model, given a significance level *alpha*.

Use `fit` to fit a model.

Pass a Numpy array to a `predict` call in order to make predictions given a model.  The method considers whether the regressors values passed to the method are 'within sample' before returning predictions.  By default, predictions are only returned if all the regressor values are 'within sample'.
