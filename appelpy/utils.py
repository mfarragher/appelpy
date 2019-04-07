import pandas as pd
import numpy as np
import itertools
__all__ = ['DummyEncoder', 'InteractionEncoder']


class DummyEncoder:
    """Encode categorical columns into dummy variables - a range of
    formats are supported.

    Steps for encoding:
    - Initialize an encoder object
    - Use the encode method from the encoder object

    Args:
        - df (pd.DataFrame): the dataframe with the categorical columns
            to encode.
        - separator (str): defaults to '_'.  The character that separates
            the category name and the category value in the encoded column.

    Method(s):
        encode: return a processed dataframe that has the categories
            encoded in the desired format.

    Raises:
        ValueError: separator argument must not be '#'.
        ValueError: nan_policy must be one of the three valid options.

    Attributes:
        categorical_col_base_levels (dict): each categorical column (key)
            and its base level (value).
        df (pd.DataFrame): the dataframe subject to the encoding.
        nan_policy (str): the policy for how to handle categories with NaN
            values for all of the categories specified in the object.
        separator (str): the separator between the category name and category
            value in all encoded columns.
    """

    def __init__(self, df, separator='_'):
        "Initializes the DummyEncoder object."
        if separator == '#':
            raise ValueError(
                """'#' is reserved for interaction terms.
                Use a different character.""")

        # Inputs for encoding:
        self._nan_policy = None
        self._df = df
        self._separator = separator
        # Outputs from encoding:
        self._categorical_col_base_levels = None

    @property
    def nan_policy(self):
        "str: NaN policy used for encoding."
        return self._nan_policy

    @property
    def df(self):
        "pd.DataFrame: dataframe to use for encoding."
        return self._df

    @property
    def separator(self):
        "str: separator character."
        return self._separator

    @property
    def categorical_col_base_levels(self):
        "dict: categorical columns paired with base levels."
        return self._categorical_col_base_levels

    def encode(self, category_base_dict, nan_policy='row_of_zero'):
        """Encode categories into dummy columns for a new dataframe.

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
            category_base_dict (dict): A dictionary comprising categories
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

        Raises:
            ValueError: nan_policy must be one of the three valid options.

        Returns:
            pd.DataFrame: dataframe with categories encoded into dummy columns.
        """

        if nan_policy not in ['row_of_zero', 'dummy_for_nan', 'row_of_nan']:
            raise ValueError("The argument for nan_policy is not valid.")

        self._categorical_col_base_levels = category_base_dict
        self._nan_policy = nan_policy

        # Initialize the dataframe that will be returned
        processed_df = self._df.copy()

        # Iterate the categorization through each col:
        for col in category_base_dict.keys():
            # Determine the base level
            base_level = category_base_dict[col]
            if category_base_dict[col] == min:
                base_level = min(self._df[col].unique().dropna())
            if category_base_dict[col] == max:
                base_level = max(self._df[col].unique().dropna())
            self._categorical_col_base_levels[col] = base_level

            # GENERATE DUMMIES GIVEN THE nan_policy
            if nan_policy == 'row_of_zero':  # Pandas default behaviour
                dummy_cols = pd.get_dummies(
                    self._df[col], prefix=col, prefix_sep=self._separator)
            if nan_policy == 'dummy_for_nan':
                # If there are no NaN category values then do not create
                # a NaN dummy:
                if np.count_nonzero((~np.isnan(self._df[col].to_numpy()))
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
            if nan_policy == 'row_of_nan':
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
                base_level_col_str = ''.join([col, self._separator, str(base_level)])
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
    - Initialize an encoder object
    - Use the encode method from the encoder object

    Method(s):
        encode: return a processed dataframe that has the categories
            encoded in the desired format.

    Attributes:
        columns_removed (list): list of columns that are in the original
            dataframe but dropped from the final dataframe.  These are
            typically the columns that represent main effects for
            categorical variables.
        columns_added (list): list of columns that are in the final dataframe
            but not in the original dataframe.  These include the column names
            for the interaction effects and the dummy columns encoded from
            categorical variables.
        df (pd.DataFrame): dataframe with columns encoded to capture
            interaction effects between the specified variables.
        separator (str): the separator between the category name and category
            value in all encoded columns, where DummyEncoder is used.
    """

    def __init__(self, df, separator='_'):
        "Initializes the InteractionEncoder object."
        # Inputs for encoding:
        self._df = df
        self._separator = separator
        # Outputs from encoding:
        self._columns_removed = None
        self._columns_added = None

    @property
    def df(self):
        "pd.DataFrame: dataframe to use for encoding."
        return self._df

    @property
    def separator(self):
        "str: separator character"
        return self._separator

    @property
    def columns_removed(self):
        "list: columns removed from the original df as a result of encoding."
        return self._columns_removed

    @property
    def columns_added(self):
        "list: columns added to the final dataframe as a result of encoding."
        return self._columns_added

    def encode(self, interactions_dict):
        """Encode interactions between variables.

        Pass a dictionary of key-value pairs, where each value is a list of
        columns to interact with the column represented by the key.

        Args:
            interactions_dict (dict): a dictionary that specifies the columns
                to be interacted.  Each dictionary value must be a list of
                column(s) in the dataframe.
                For example, {'temperature': ['pressure']} would
                be the specification to generate an interaction effect
                term 'temperature#pressure' for an interaction between the two
                continuous variables 'temperature' and 'pressure'.

        Returns:
            pd.DataFrame: dataframe with interaction effects added.
        """

        # Initialize dataframe
        processed_df = self._df.copy()

        # Iterate the generation of interaction cols through each pair of cols:
        for col_name, cols_to_interact_list in interactions_dict.items():
            for col_other in cols_to_interact_list:
                # Gather dtype strings for the cols, e.g. 'categorical'
                col_dtype = self._df[col_name].dtype.name
                col_other_dtype = self._df[col_other].dtype.name

                # Truth conditions for whether each column is Boolean
                col_bool = self._df[col_name].isin([0, 1, np.NaN]).all()  # all non-NaN Series vals are in {0, 1}
                col_other_bool = self._df[col_other].isin([0, 1, np.NaN]).all()

                # Create dummy columns from the original columns
                if col_bool:
                    # Use original
                    col_dummies = pd.Series(self._df[col_name], name=col_name)
                if col_dtype == 'category':
                    col_dummies = pd.get_dummies(self._df[col_name], prefix=str(col_name))
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
                    interaction_term = "#".join(str(col) for col in (col_name, col_other))
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
                    interaction_term = "#".join(str(col) for col in (col_name, col_other))
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
                                                 separator=self._separator)
                    processed_df = dummy_encoder.encode({col_name: None, col_other: None})
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
                                                 separator=self._separator)
                    processed_df = dummy_encoder.encode({col_other: None})
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
                                                 separator=self._separator)
                    processed_df = dummy_encoder.encode({col_name: None})
                    # Add interaction cols:
                    processed_df = pd.concat([processed_df, interaction_dummies_df],
                                             axis='columns')
                    continue

                # One Boolean col and one continuous col:
                # 6)
                if col_bool and col_other_dtype != 'category':
                    interaction_term = "#".join(str(col) for col in (col_name, col_other))
                    interaction_dummy = pd.Series(col_dummies * self._df[col_other],
                                                  name=interaction_term)
                    processed_df = pd.concat([processed_df, interaction_dummy],
                                             axis='columns')
                    continue
                # 7)
                if col_dtype != 'category' and col_other_bool:
                    interaction_term = "#".join(str(col) for col in (col_name, col_other))
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
                        interaction_term = "#".join(str(col) for col in (col_value, col_other))
                        interaction_dummies_df[interaction_term] = (col_dummies[col_value]
                                                                    * self._df[col_other])
                    # Remove original non-categorical cols and replace with dummies:
                    dummy_encoder = DummyEncoder(processed_df,
                                                 separator=self._separator)
                    processed_df = dummy_encoder.encode({col_name: None})
                    # Add interaction cols:
                    processed_df = pd.concat([processed_df, interaction_dummies_df],
                                             axis='columns')
                    continue
                # 9)
                if (col_dtype != 'category' and not col_bool) and col_other_dtype == 'category':
                    for combo in itertools.product([col_name], col_other_dummies.columns.values):
                        col_name, col_other_value = combo
                        interaction_term = "#".join(str(col) for col in (col_name, col_other_value))
                        interaction_dummies_df[interaction_term] = (col_other_dummies[col_other_value]
                                                                    * self._df[col_name])
                    # Remove original non-categorical cols and replace with dummies:
                    dummy_encoder = DummyEncoder(processed_df,
                                                 separator=self._separator)
                    processed_df = dummy_encoder.encode({col_other: None})
                    # Add interaction cols:
                    processed_df = pd.concat([processed_df, interaction_dummies_df],
                                             axis='columns')
                    continue

        # Collect columns added to the final dataframe and removed from original dataframe:
        self._columns_removed = list((set(self._df.columns)
                                      - set(processed_df.columns)))
        self._columns_added = list((set(processed_df.columns)
                                    - set(self._df.columns)))

        return processed_df
