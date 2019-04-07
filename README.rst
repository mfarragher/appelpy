===================================================
appelpy: Applied Econometrics Library for Python
===================================================

*********
About 👁️
*********
**appelpy** is the *Applied Econometrics Library for Python*.  It seeks to bridge the gap between the software options that have a simple syntax (such as Stata) and other powerful options that use Python's object-oriented programming as part of data modelling workflows.  ⚗️

Econometric modelling and general regression analysis in Python have never been easier!

The library builds upon the functionality of the 'vanilla' Python data stack (e.g. Pandas, Numpy, etc.) and other libraries such as Statsmodels.

See the `appelpy-examples <https://github.com/mfarragher/appelpy-examples>`_ Github repo for more detailed **documentation and notebooks** that show the functionality of the library.


🥧 Why it's as easy as pie
==========================
Here is a flavour of a basic OLS regression done through appelpy, supposing you have `data <https://econpapers.repec.org/paper/bocbocins/caschool.htm>`_ sitting in a Pandas dataframe ``df`` and want to model the dependent variable ``api00`` on three other variables:::

        from appelpy.linear_model import OLS
        model1 = OLS(df, ['api00'], ['acs_k3', 'meals', 'full'])
        model1.results_output  # returns summary results

The key information is sitting in the ``model1`` object, but there is much more functionality that can be done with it.  These are more things that can be done via one line of code:

* *Diagnostics* can be called from the object: e.g. produce a P-P plot via ``model1.diagnostic_plot('pp_plot')``
* *Model selection statistics*: e.g. find the root mean square error of the model from ``model1.model_selection_stats``
* *Standardized model estimates*: ``model1.results_output_standardized``


🍏 What inspired it?
====================
1) The simple syntax of software such as Stata.  With the data loaded, a regression model summary can be returned by a one-line command:

    .. code-block:: stata
 
       regress api00 acs_k3 meals full

  However with the simplicity comes a few disadvantages: it is not open-source software; the workflows are tricky with modern business problems; lacks the benefits of object-oriented programming.

2) Statsmodels is a powerful Python library that addresses some of those disadvantages, but with that power comes a considerable learning curve and clunkiness.  Here is the code for the same regression:::

        import statsmodels.api as sm
        model1 = sm.OLS(df['api00'], sm.add_constant(df['acs_k3', 'meals', 'full'])).fit()
        results1 = model1.summary()  # returns summary results

  It can get much more unwieldy than that.  The model results object is brilliant as it can be printed in different formats (plaintext, Latex, etc.)... but that is only the starting point.  How do I diagnose the regression model itself?  How do I get standardized estimates?  That's where it becomes more complicated.

**appelpy** simply wants to achieve a *sweet spot* between both approaches.


*****************
Installation ⏲️
*****************
``pip install appelpy``


******************
Dependencies 🖇️
******************
- pandas>=0.24
- scipy
- numpy
- statsmodels>=0.8
- patsy
- seaborn
- matplotlib


*************
Licence ⚖️
*************
Modified BSD (3-clause)