import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import PolynomialFeatures

def one_hot_dataframe(data,cols,replace=False):
    """

    Args:
        data: input data frame need to encode categorical feature
        cols: list of columns that needed to be encode
        replace: replace the data frame?

    Returns: 3-tuple comprising the data, vectorized data, and fitted vectorizor

    """
    vec = DictVectorizer(sparse=False)
    vecData = pd.DataFrame(vec.fit_transform(data[cols].T.to_dict().values()))
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data, vecData, vec)


def scaling_feature(df, meta_df, *args, replace = False):
    """

    Args:
        df: dataframe with features to scale
        meta_df: meta data contains range of features
        *args: list of feature needed to be scale

    Returns: dataframe after scaling features (if replace = True)

    """
    _df = df.copy()
    for feature in args:
        # print(feature)
        f = lambda x: x[feature]/meta_df.ix[feature, 'Max']
        _df[feature] = _df.apply(f, axis=1)
        # print(_df)
    if replace is True:
        df = _df.copy()
    return _df


def encode_final_score(df, target_column, replace=False):
    """Add column to df with integers for the target.

        Args
        ----
        df -- pandas DataFrame.
        target_column -- column to map to int, producing
                         new Target column.

        Returns
        -------
        df_mod -- modified DataFrame.
        targets -- list of target names.
    """
    df_mod = df.copy()
    for index, row in df_mod.iterrows():
        cur = row[target_column]
        if cur <= 5: df_mod.loc[index, target_column] = int(1)
        elif cur <= 10: df_mod.loc[index, target_column] = int(2)
        elif cur <= 15: df_mod.loc[index, target_column] = int(3)
        elif cur <= 20: df_mod.loc[index, target_column] = int(4)
        # elif cur >= 50: df_mod.loc[index, target_column] = 5
        else: df_mod.loc[index, target_column] = int(5)
    if replace is True:
        df = df_mod.copy()
    return df_mod, df


def add_polynomial_features(input_df, *poly_features, degree=2):
    """add polynomial features

    Args:
        input_df: original DataFrame
        *poly_features: features to add polynomial terms
        degree: degree of polynomial

    Returns:
        DataFrame after adding polynomial features

    """
    non_poly = input_df.ix[:, [name for name in list(input_df.columns.values)]]
    poly_df = input_df.ix[:, poly_features]
    poly = PolynomialFeatures(degree)
    output_nparray = poly.fit_transform(poly_df)

    target_feature_names = ['x'.join(['{}^{}'.format(pair[0], pair[1]) for pair in tuple if pair[1] != 0]) for tuple in
                            [zip(poly_df.columns, p) for p in poly.powers_]]
    output_df = pd.DataFrame(output_nparray, columns=target_feature_names)

    return pd.concat([non_poly, output_df], axis=1)


