<header>
<pre><p style="font-size:28px;"><b>utils</b></p></pre>
</header>

# Overview
The **`utils`** module has classes and functions to support pre-processing of datasets before modelling.

The main classes are:

- `DummyEncoder`
- `InteractionEncoder`

## `DummyEncoder`
[![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/mfarragher/appelpy-examples/blob/master/01-01_dummy-and-interaction-encoders_hsbdemo.ipynb): static render of a notebook that has an example of dummy variable encoding.

Suppose `schtyp`, `prog` and `honors` are categorical variables in a dataframe `df_raw` and we want to make dummy variables for them in a new transformed dataset.  A new dataframe `df` is returned by calling the `transform` method on an instance of `DummyEncoder`:
```python
from appelpy.utils import DummyEncoder
df = (DummyEncoder(df_raw, {'schtyp': None,
                            'prog': None,
                            'honors': None})
      .transform())
```

The class is initialised with:

- A raw dataframe
- A dictionary of key-value pairs, where each pair has a categorical variable as a key and a base level as a value.
- A 'NaN policy' to detemine how NaN values should be treated in a dataset, if any.  Where a categorical variable has a NaN value, the default behaviour makes the dummy columns have values of 0.  The keyword argument can be changed to deal with cases where 'missingness' of data is not random, e.g. create a column for NaN value or make all dummy columns have NaN value.

If the base level is None then a dummy column is created for every value of a categorical variable.

## `InteractionEncoder`
[![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/mfarragher/appelpy-examples/blob/master/01-01_dummy-and-interaction-encoders_hsbdemo.ipynb): static render of a notebook that shows the many ways in which interaction effects can be encoded (e.g. between two Boolean variables, a continuous variable & categorical variable, etc.).

Suppose `math` and `socst` are two continuous variables in a dataframe `df_raw` and we want to make an interaction effect `math#socst`.  A new dataframe `df_model` is returned – with the new column `math#socst` – by calling the `transform` method on an instance of `InteractionEncoder`:
```python
from appelpy.utils import InteractionEncoder
df_model = InteractionEncoder(df_raw, {'math': ['socst']}).transform()
```

## `get_dataframe_columns_diff` method
This method compares the columns in two dataframes, so it is handy when comparing a raw dataframe and a transformed dataframe.

Suppose that there is a raw dataframe `df_raw` and a transformed dataframe `df_enc`.  The recipe below will display the columns removed from the raw dataframe and the columns added to the transformed dataframe:
```python
print(f"Columns removed: {get_dataframe_columns_diff(df_raw, df_enc)}")
print(f"Columns added: {get_dataframe_columns_diff(df_enc, df_raw)}")
```
