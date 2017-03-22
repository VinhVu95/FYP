import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel


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


def remove_low_variance_features(input_df, thres=0):
    sel = VarianceThreshold(threshold=thres)
    sel.fit(input_df)
    index = np.where(sel.variances_ > thres)[0]
    return input_df.ix[:, index]


def feature_importance(x, y, transform=False, th=None):
    """ select the most relevant features to fit into model

    Args:
        x: input features
        y: output to predict by the model
        transfrom: reduce x to the selected features
        threshold: The threshold value to use for feature selection. Features whose importance is greater or
                   equal are kept while the others are discarded. If “median” (resp. “mean”), then the
                   threshold value is the median (resp. the mean) of the feature importances. A scaling
                   factor (e.g., “1.25*mean”) may also be used. If None and if the estimator has a parameter
                   penalty set to l1, either explicitly or implicitly (e.g, Lasso), the threshold used is 1e-5.
                   Otherwise, “mean” is used by default.
    Returns:
        Array of feature importance using ExtraTreeClassifier

    """
    clf = ExtraTreesRegressor(min_samples_split=3, min_samples_leaf=3, max_depth=5)
    clf = clf.fit(x, y)
    if transform == True:
        model = SelectFromModel(clf, threshold=th, prefit=True)
        model.transform(x)
    importances = clf.feature_importances_
    print("Feature ranking:")
    indices = np.argsort(importances)[::-1]
    headers = list(x.columns.values)
    for f in range(x.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, headers[indices[f]], importances[indices[f]]))

    return importances


def extract_test_set(input, output, *extracted_row_index):
    """

    Args:
        input: data frame to extract test input attributes
        output: serie to extract test output attribute
        *extracted_row_index: list of index assigned as test object in original datasets

    Returns: test_set_input: test input data frame
             test_set_out_put: test output serie

    """
    test_set_input = input.ix[extracted_row_index]
    test_set_output = output.ix[extracted_row_index]
    input.drop(input.index[extracted_row_index], inplace=True)
    input.reset_index(inplace=True)
    input.drop('index', axis=1, inplace=True)
    output.drop(output.index[extracted_row_index], inplace=True)
    if isinstance(output, pd.DataFrame):
        output.reset_index(inplace=True)
        input.drop('index', axis=1, inplace=True)
    else:
        output.reset_index(drop=True, inplace=True)

    return test_set_input, test_set_output

