<p style="font-size:28px;"><b>appelpy: Applied Econometrics Library for Python</b></p>

**appelpy** is the *Applied Econometrics Library for Python*.  It seeks to bridge the gap between the software options that have a simple syntax (such as Stata) and other powerful options that use Python's object-oriented programming as part of data modelling workflows.  ⚗️

Econometric modelling and general regression analysis in Python have never been easier!

The library builds upon the functionality of the 'vanilla' Python data stack (e.g. Pandas, Numpy, etc.) and other libraries such as Statsmodels.

## 10 Minutes to Appelpy
Explore the core functionality of Appelpy in the **10 Minutes To Appelpy** notebook (click the badges):

- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mfarragher/appelpy-examples/master?filepath=00_ten-minutes-to-appelpy.ipynb): interactive experience of the *10 Minutes to Appelpy* tutorial via Binder.
- [![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/mfarragher/appelpy-examples/blob/master/00_ten-minutes-to-appelpy.ipynb): static render of the *10 Minutes to Appelpy* notebook.

# Installation
Install the library via the Pip command:
``` bash
pip install appelpy
```

Supported for Python 3.6 and higher versions.

# Why Appelpy?
## Basic usage
It only takes **one line of code** to fit a basic linear model of 'y on X' and another line to return the model's results.

```python
from appelpy.linear_model import OLS

model1 = OLS(df, y_list, X_list).fit()  # y_list & X_list contain df columns
model1.results_output  # returns (Statsmodels) summary results
```

Model objects have many useful attributes, e.g. the inputs X & y, standardized X and y values, results of fitted models (incl. standardized estimates).  The library also has **diagnostic classes and functions** that consume model objects (or else their underlying data).

These are more things that can be obtained via **one line of code:**

* *Diagnostics* can be called from the object: e.g. produce a P-P plot via `model1.diagnostic_plot('pp_plot')`
* *Model selection statistics*: e.g. find the root mean square error of the model from `model1.model_selection_stats`
* *Standardized model estimates*: `model1.results_output_standardized`

Classes in the library have a fluent interface, so that they can be instantiated and have chained methods in one line of code.

## Features that add value to model workflows in Python
See Appelpy's **[key features](intro/key-features.md)** (with images), which add _so much more_ to the vanilla Python data stack, e.g.:

- Fluent interface and API design make it easier to build pipelines for modelling & data pre-processing.
- More accessible interface for Stata users, while utilising the benefits of object-orientated programming.
- One method for calling **diagnostic plots** to assess whether OLS assumptions hold in a model.
- **Useful encoders** for transforming datasets, e.g. `DummyEncoder` and `InteractionEncoder`.
- Standardized model estimates (Beta coefficients).
- Decomposition of influence analysis into three parts: leverage, outlier and influence measures.
- Identify extreme observations in a model based on common heuristics.
- **Perform diagnostics not implemented in the main Python libraries**, e.g. studentized Breusch–Pagan test of heteroskedasticity.

# Modules
## Exploration and pre-processing
- **`eda`:** functions for exploratory data analysis (EDA) of datasets, e.g. `statistical_moments` for obtaining mean, variance, skewness and kurtosis of all numeric columns.
- **`utils`:** classes and functions for data pre-processing, e.g. encoding of interaction effects and dummy variables in datasets.
    - `DummyEncoder`: encode dummy variables in a dataset based on different policies for dealing with NaN values.
    - `InteractionEncoder`: encode interaction effects of variables in a dataset.
## Model fitting
- **`linear_model`:** classes for linear models such as Ordinary Least Squares (OLS) and Weighted Least Squares (WLS).
- **`discrete_model`:** classes for discrete choice models, e.g. logistic regression (Logit).
## Model diagnostics
- **`diagnostics`:**
    - `BadApples`: class for inspecting observations that could 'stink up' a model, i.e. the observations that are outliers, high-leverage points or else have high influence in a model.
    - `variance_inflation_factors`: function that returns variance inflation factor (VIF) scores for regressors in a dataset.
    - `partial_regression_plot`: also known as 'added variable plot'.  Examine the effect of adding a regressor to a model.
