<header>
<pre><p style="font-size:28px;"><b>eda</b></p></pre>
</header>

# Overview
The **`eda`** module has functions to support exploratory data analysis.

## Statistical moments
The **10 Minutes To Appelpy** notebook shows the statistical moments of the California Test Score dataset.

- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mfarragher/appelpy-examples/master?filepath=00_ten-minutes-to-appelpy.ipynb): interactive experience of the *10 Minutes to Appelpy* tutorial via Binder.
- [![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/mfarragher/appelpy-examples/blob/master/00_ten-minutes-to-appelpy.ipynb): static render of the *10 Minutes to Appelpy* notebook.

```python
from appelpy.eda import statistical_moments
statistical_moments(df)
```

## Correlation heatmap
The `correlation_heatmap` method produces a heatmap (triangular form) of the correlation matrix, given a dataset `df`.
