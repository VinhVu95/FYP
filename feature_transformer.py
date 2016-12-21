import pandas as pd
from sklearn.feature_extraction import DictVectorizer

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
