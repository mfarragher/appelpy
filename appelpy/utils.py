import sys
import os
import pandas as pd
import numpy as np
import itertools
__all__ = ['DummyEncoder', 'InteractionEncoder',
           'get_dataframe_columns_diff']


class _SuppressPrints:
    """For internal use to suppress print output from Statsmodels that
    is not wanted in Appelpy.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def _df_input_conditions(X, y):
    if (y.isin([np.inf, -np.inf]).any() or
            len(X[X.isin([np.inf, -np.inf]).any(1)]) > 0):
        raise ValueError(
            '''Remove infinite (positive or negative) values from the
            dataset before modelling.''')

    if (len(X.select_dtypes(['category']).columns.tolist()) > 0
        or
            pd.api.types.is_categorical_dtype(y.dtype)):
        raise TypeError(
            '''Encode dummies from Pandas Category column(s) before
            modelling.''')

    if (len(X.select_dtypes(['O']).columns.tolist()) > 0 or
            pd.api.types.is_string_dtype(y.dtype)):
        raise TypeError(
            '''Remove columns with string data before modelling.  If
            categorical data, then use their dummy columns in model.''')

    if y.isnull().values.any() or X.isnull().values.any():
        raise ValueError(
            '''Remove observations with NaN values before modelling.''')
    pass


class DummyEncoder:
    """Encode categorical columns into dummy variables - a range of
    formats are supported.

    Steps for encoding:
    - Initialize an encoder object (with source dataframe, base level info and
        policy)
    - Call the transform method and assign the result to a dataframe

    A 'base level' can be specified for a category (via a
    dictionary) so that the dummy column for one category value can
    be dropped.  For example, if category_base_dict is {'rank': 1}
    then the dummy column for rank_1 is dropped after the encoding.

    There are three different policies for dealing with NaN values
    in a category:
    - row_of_zero: the default behaviour in Pandas get_dummies.
        If the category has a NaN value for a given row, then dummy
        columns are made from the non-NaN values and so every dummy
        value will be 0 across the row.
    - dummy_for_nan: A dummy column is created to signify NaN values
        for the category (if there are any NaN values for the category).
    - row_of_nan: If the category has a NaN value for a given row, then
        dummy columns are made from the non-NaN values, but every dummy
        value will be NaN across the row.  This is typically the best
        policy for cases where the NaN values are due to skip patterns
        or data are missing by design.  For example, if someone is not
        asked (by design) a set of questions in a survey then the values
        of those questions would be best represented by NaN.

    Args:
        df (pd.DataFrame): the dataframe with the categorical columns
            to encode.
        separator (str): defaults to '_'.  The character that separates
            the category name and the category value in the encoded column.
        categorical_col_base_levels (dict): A dictionary comprising categories
            paired to a base level, e.g. {'rank': 1, 'country':
            'US', 'age': min}.  The dummy column for the base level
            is dropped.  If a base level is not desired then
            specify None explicitly.  The built-in functions max
            and min can be specified as base levels for the highest
            and lowest values (respectively) of a category.
        nan_policy (str, optional): Defaults to 'row_of_zero'.  Select
            one of three policies for encoding the dummy columns:
            'row_of_zero', 'dummy_for_nan' and 'row_of_nan'.  See the main
            description above for an explanation of how each nan_policy
            works.

    Method(s):
        transform: return a processed dataframe that has the categories
            encoded in the desired format.

    Raises:
        ValueError: separator argument must not be '#'.
        ValueError: nan_policy argument must be one of the three specified
            in Args.

    Attributes:
        df (pd.DataFrame): the dataframe subject to the encoding.
        categorical_col_base_levels (dict): the argument passed to the object.
        separator (str): the separator between the category name and category
            value in all encoded columns.
    """

    def __init__(self, df, categorical_col_base_levels,
                 nan_policy='row_of_zero', separator='_'):
        "Initializes the DummyEncoder object."
        if separator == '#':
            raise ValueError(
                """'#' is reserved for interaction terms.
                Use a different character.""")
        if nan_policy not in ['row_of_zero', 'dummy_for_nan', 'row_of_nan']:
            raise ValueError("The argument for nan_policy is not valid.")

        # Inputs for encoding:
        self._df = df
        self._categorical_col_base_levels = categorical_col_base_levels
        self._nan_policy = nan_policy
        self._separator = separator

    @property
    def df(self):
        "pd.DataFrame: dataframe to use for encoding."
        return self._df

    @property
    def categorical_col_base_levels(self):
        "dict: categorical columns paired with base levels."
        return self._categorical_col_base_levels

    @property
    def nan_policy(self):
        "str: NaN policy used for encoding."
        return self._nan_policy

    @property
    def separator(self):
        "str: separator character."
        return self._separator

    def transform(self):
        """Encode categories into dummy columns for a new dataframe.

        Returns:
            pd.DataFrame: dataframe with categories encoded into dummy columns.
        """
        # Initialize the dataframe that will be returned
        processed_df = self._df.copy()

        # Iterate the categorization through each col:
        for col in self._categorical_col_base_levels.keys():
            # Determine the base level
            base_level = self._categorical_col_base_levels[col]
            if self._categorical_col_base_levels[col] == min:
                base_level = min(self._df[col].dropna().unique())
                self._categorical_col_base_levels[col] = base_level
            if self._categorical_col_base_levels[col] == max:
                base_level = max(self._df[col].dropna().unique())
                self._categorical_col_base_levels[col] = base_level

            # GENERATE DUMMIES GIVEN THE nan_policy
            if self._nan_policy == 'row_of_zero':  # Pandas default behaviour
                dummy_cols = pd.get_dummies(
                    self._df[col], prefix=col, prefix_sep=self._separator)
            if self._nan_policy == 'dummy_for_nan':
                # If there are no NaN category values then do not create
                # a NaN dummy:
                if np.count_nonzero((~pd.isna(self._df[col].to_numpy()))
                                    == len(self._df[col].to_numpy())):
                    dummy_cols = pd.get_dummies(
                        self._df[col], prefix=col, prefix_sep=self._separator)
                else:
                    # Note: pd.get_dummies(... dummy_na=True) is not robust
                    # for nullable Int Series
                    dummy_cols = pd.get_dummies(
                        self._df[col], prefix=col, prefix_sep=self._separator)
                    # Create NaN dummy given values of the other dummies:
                    nan_dummy_col_str = ''.join([col, self._separator, 'nan'])
                    dummy_cols[nan_dummy_col_str] = np.where(
                        dummy_cols.sum(axis='columns') == 0, 1, 0)
            if self._nan_policy == 'row_of_nan':
                dummy_cols = pd.get_dummies(self._df[col], prefix=col,
                                            prefix_sep=self._separator)
                # Replace the zero vals with NaN:
                if self._df[col].isna().any():
                    nan_row_indices = list(dummy_cols[((dummy_cols == 0)
                                                       .all(axis='columns'))].index)
                    dummy_cols.loc[nan_row_indices] = np.NaN
                    # Make sure dummy columns are nullable Int after setting those NaN values
                    dummy_cols = dummy_cols.astype(pd.Int64Dtype())

            # Remove base
            if base_level is not None:  # 'is not None' will allow 0, 0.0 to pass
                base_level_col_str = ''.join(
                    [col, self._separator, str(base_level)])
                del dummy_cols[base_level_col_str]
            # Concat dataframe and dummies
            processed_df = pd.concat([processed_df, dummy_cols],
                                     axis='columns')
            # Remove original col
            del processed_df[col]
        return processed_df


class InteractionEncoder:
    """Encoder for interaction effects between variables.

    An encoded column for the interaction effect between two variables
    will have a '#' separator between the names of the variables.
    For example, the interaction between 'temperature' and 'pressure'
    variables will be 'temperature#pressure'.

    These are examples of interactions between variables, which are handled
    by the encoder:
    - Two Boolean variables
    - Two continuous variables
    - Two categorical variables
    - One Boolean and one categorical variable
    - One Boolean and one continuous variable
    - One categorical and one continuous variable

    NOTE: categorical variables for the encoder are supported by the dtype
    pd.Categorical.

    Steps for encoding:
    - Initialize an encoder object (with source dataframe and dictionary
        of columns to interact)
    - Call the transform method and assign the result to a dataframe

    Method(s):
        transform: return a processed dataframe that has the categories
            encoded in the desired format.

    Attributes:
        df (pd.DataFrame): dataframe with columns encoded to capture
            interaction effects between the specified variables.
        interactions (dict): a dictionary that specifies the columns
            to be interacted.  Each dictionary value must be a list of
            column(s) in the dataframe.
            For example, {'temperature': ['pressure']} would
            be the specification to generate an interaction effect
            term 'temperature#pressure' for an interaction between the two
            continuous variables 'temperature' and 'pressure'.
        separator (str): the separator between the category name and category
            value in all encoded columns, where DummyEncoder is used.
    """

    def __init__(self, df, interactions, separator='_'):
        "Initializes the InteractionEncoder object."
        # Inputs for encoding:
        self._df = df
        self._interactions = interactions
        self._separator = separator

    @property
    def df(self):
        "pd.DataFrame: dataframe to use for encoding."
        return self._df

    @property
    def interactions(self):
        "dict: the columns to be interacted."
        return self._interactions

    @property
    def separator(self):
        "str: separator character"
        return self._separator

    def transform(self):
        """Encode interactions between variables.

        Returns:
            pd.DataFrame: dataframe with interaction effects added.
        """

        # Initialize dataframe
        processed_df = self._df.copy()

        # Iterate the generation of interaction cols through each pair of cols:
        for col_name, cols_to_interact_list in self._interactions.items():
            for col_other in cols_to_interact_list:
                # Gather dtype strings for the cols, e.g. 'categorical'
                col_dtype = self._df[col_name].dtype.name
                col_other_dtype = self._df[col_other].dtype.name

                # Truth conditions for whether each column is Boolean
                # all non-NaN Series vals are in {0, 1}
                col_bool = self._df[col_name].isin([0, 1, np.NaN]).all()
                col_other_bool = self._df[col_other].isin([0, 1, np.NaN]).all()

                # Create dummy columns from the original columns
                if col_bool:
                    # Use original
                    col_dummies = pd.Series(self._df[col_name], name=col_name)
                if col_dtype == 'category':
                    col_dummies = pd.get_dummies(
                        self._df[col_name], prefix=str(col_name))
                if col_other_bool:
                    col_other_dummies = pd.Series(self._df[col_other],
                                                  name=col_other)
                if col_other_dtype == 'category':
                    col_other_dummies = pd.get_dummies(self._df[col_other],
                                                       prefix=str(col_other))

                # CASES FOR INTERACTION TERM ENCODING:
                # 1) both cols are Boolean
                # 2) both cols are continuous
                # 3) both cols are categorical
                # 4) & 5) one col is Boolean and the other is categorical
                # 6) & 7) one col is Boolean and the other is continuous
                # 8) & 9) one col is categorical and the other is continuous

                # 1) Both cols are Boolean:
                # Straightforward case - add to df one column that multiplies both cols
                if col_bool and col_other_bool:
                    interaction_term = "#".join(str(col)
                                                for col in (col_name, col_other))
                    interaction_dummy = pd.Series(col_dummies * col_other_dummies,
                                                  name=interaction_term,
                                                  dtype=pd.Int64Dtype())
                    processed_df = pd.concat([processed_df, interaction_dummy],
                                             axis='columns')
                    continue

                # 2) Both cols are continuous:
                # Straightforward case - add to df one column that multiplies both cols
                if ((col_dtype != 'category' and not col_bool)
                        and (col_other_dtype != 'category' and not col_other_bool)):
                    interaction_term = "#".join(str(col)
                                                for col in (col_name, col_other))
                    processed_df[interaction_term] = (processed_df[col_name]
                                                      * processed_df[col_other])
                    continue

                # Set up auxiliary dataframe to store all the dummy columns
                if ((col_bool or col_dtype == 'category')
                        and (col_other_bool or col_other_dtype == 'category')):
                    aux_df = pd.concat([col_dummies, col_other_dummies],
                                       axis='columns')
                # Set up structure to store the interaction variables
                interaction_dummies_df = pd.DataFrame(index=self._df.index)
                interaction_terms_list = []  # will contain e.g. 'col_val1#col_other_val1'

                # 3) Both cols are categorical:
                if col_dtype == 'category' and col_other_dtype == 'category':
                    for combo in itertools.product(col_dummies, col_other_dummies):
                        col_value, col_other_value = combo
                        interaction_term = "#".join(str(col) for col in combo)
                        interaction_terms_list.append(interaction_term)
                        interaction_dummies_df[interaction_term] = (aux_df[col_value] *
                                                                    aux_df[col_other_value])
                    # Remove original non-categorical cols and replace with dummies:
                    dummy_encoder = DummyEncoder(processed_df,
                                                 {col_name: None,
                                                  col_other: None},
                                                 separator=self._separator)
                    processed_df = dummy_encoder.transform()
                    # Add interaction cols:
                    processed_df = pd.concat([processed_df, interaction_dummies_df],
                                             axis='columns')
                    continue

                # One Boolean col and one categorical col:
                # 4)
                if col_bool and col_other_dtype == 'category':
                    for combo in itertools.product([col_name], col_other_dummies.columns.values):
                        col_name, col_other_value = combo
                        interaction_term = "#".join(str(col) for col in combo)
                        interaction_terms_list.append(interaction_term)
                        interaction_dummies_df[interaction_term] = (col_dummies *
                                                                    aux_df[col_other_value])
                    # Remove original non-Bool col and replace with dummies:
                    dummy_encoder = DummyEncoder(processed_df,
                                                 {col_other: None},
                                                 separator=self._separator)
                    processed_df = dummy_encoder.transform()
                    # Add interaction cols:
                    processed_df = pd.concat([processed_df, interaction_dummies_df],
                                             axis='columns')
                    continue
                # 5)
                if col_dtype == 'category' and col_other_bool:
                    for combo in itertools.product(col_dummies.columns.values, [col_other]):
                        col_value, col_other = combo
                        interaction_term = "#".join(str(col) for col in combo)
                        interaction_terms_list.append(interaction_term)
                        interaction_dummies_df[interaction_term] = (aux_df[col_value]
                                                                    * col_other_dummies)
                    # Remove original non-Bool col and replace with dummies:
                    dummy_encoder = DummyEncoder(processed_df,
                                                 {col_name: None},
                                                 separator=self._separator)
                    processed_df = dummy_encoder.transform()
                    # Add interaction cols:
                    processed_df = pd.concat([processed_df, interaction_dummies_df],
                                             axis='columns')
                    continue

                # One Boolean col and one continuous col:
                # 6)
                if col_bool and col_other_dtype != 'category':
                    interaction_term = "#".join(str(col)
                                                for col in (col_name, col_other))
                    interaction_dummy = pd.Series(col_dummies * self._df[col_other],
                                                  name=interaction_term)
                    processed_df = pd.concat([processed_df, interaction_dummy],
                                             axis='columns')
                    continue
                # 7)
                if col_dtype != 'category' and col_other_bool:
                    interaction_term = "#".join(str(col)
                                                for col in (col_name, col_other))
                    interaction_dummy = pd.Series(self._df[col_name] * col_other_dummies,
                                                  name=interaction_term)
                    processed_df = pd.concat([processed_df, interaction_dummy],
                                             axis='columns')
                    continue

                # One categorical col and one continuous col:
                # 8)
                if col_dtype == 'category' and (col_other_dtype != 'category'
                                                and not col_other_bool):
                    for combo in itertools.product(col_dummies.columns.values, [col_other]):
                        col_value, col_other = combo
                        interaction_term = "#".join(
                            str(col) for col in (col_value, col_other))
                        interaction_dummies_df[interaction_term] = (col_dummies[col_value]
                                                                    * self._df[col_other])
                    # Remove original non-categorical cols and replace with dummies:
                    dummy_encoder = DummyEncoder(processed_df,
                                                 {col_name: None},
                                                 separator=self._separator)
                    processed_df = dummy_encoder.transform()
                    # Add interaction cols:
                    processed_df = pd.concat([processed_df, interaction_dummies_df],
                                             axis='columns')
                    continue
                # 9)
                if (col_dtype != 'category' and not col_bool) and col_other_dtype == 'category':
                    for combo in itertools.product([col_name], col_other_dummies.columns.values):
                        col_name, col_other_value = combo
                        interaction_term = "#".join(
                            str(col) for col in (col_name, col_other_value))
                        interaction_dummies_df[interaction_term] = (col_other_dummies[col_other_value]
                                                                    * self._df[col_name])
                    # Remove original non-categorical cols and replace with
                    # dummies:
                    dummy_encoder = DummyEncoder(processed_df,
                                                 {col_other: None},
                                                 separator=self._separator)
                    processed_df = dummy_encoder.transform()
                    # Add interaction cols:
                    processed_df = pd.concat([processed_df,
                                              interaction_dummies_df],
                                             axis='columns')
                    continue

        return processed_df


def get_dataframe_columns_diff(df_minuend, df_subtrahend):
    """Get the diff between the columns of two dataframes, where
    df_minuend - df_subtrahend = df_diff

    This method is useful for keeping track of differences between
    dataframes' columns when doing column encoding (e.g. comparing a
    raw dataframe `df_raw` and transformed dataframe `df_transformed`).

    To get the 'columns added' to `df_raw` compared to `df_transformed`,
    call the function:
        cols_added = get_dataframe_columns_diff(df_transformed, df_raw)

    To get the 'columns removed' from `df_raw`, compared to `df_transformed`,
    call the function:
        cols_removed = get_dataframe_columns_diff(df_raw, df_transformed)

    Args:
        df_minuend (pd.DataFrame): the dataframe before the minus operator.
        df_subtrahend (pd.DataFrame): the dataframe after the minus operator.

    Raises:
        TypeError: Ensure Pandas dataframes are given as arguments.

    Returns:
        list: the diff in columns between the two dataframes.
    """
    if (not isinstance(df_minuend, pd.DataFrame) or
            (not isinstance(df_subtrahend, pd.DataFrame))):
        raise TypeError("Ensure Pandas dataframes are given as arguments.")
    return list(set(df_minuend.columns) - set(df_subtrahend.columns))
