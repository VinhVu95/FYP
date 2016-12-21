import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import model_selection_util as ms
# import feature_transformer as ft

from model_selection_util import best_config,best_model,cv_score_gen,plot_learning_curve
from feature_transformer import one_hot_dataframe, scaling_feature
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVR
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import Ridge,Lasso,ElasticNet

data = pd.read_csv("Student Results.csv")
# drop empty row from table
result = data.dropna()
# drop student did not attend final
result = result[result['Final Total'] != 'ABS']
# after deleting row with student absent in final, the index not shift back -> reindex, and then drop the column 'index'
result = result.reset_index().drop('index', axis=1)

# convert data type of marks to int and float
result.ix[:, 0] = result.ix[:, 0].astype(int)
result.ix[:, 3:7] = result.ix[:, 3:7].astype(float)
result.ix[:, 10:19] = result.ix[:, 10:19].astype(float)

# store meta data for the table
meta_data = {'Max': [100, 40, 10, 50, 20, 30, 20, 70, 100, 100, 25, 25, 25, 25, 100, 100],
             'Weightage': [4, 9.6, 2.4, 12, 3.4, 5.2, 3.4, 12, 12, 40, 15, 15, 15, 15, 60, 100.00]}
meta_frame = pd.DataFrame(meta_data, columns=['Max', 'Weightage'], index=list(result.columns.values)[3:])


# figure out outlier
def mad_based_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def percentile_based_outlier(points, threshold=95):
    diff = (100 - threshold) / 2.0
    minval, maxval = np.percentile(points, [diff, 100 - diff])
    return (points < minval) | (points > maxval)


def detect_outliers(x):
    fig, axes = plt.subplots(nrows=2)
    for ax, func in zip(axes, [percentile_based_outlier, mad_based_outlier]):
        sns.distplot(x, ax=ax, rug=True, hist=False)
        outliers = x[func(x)]
        ax.plot(outliers, np.zeros_like(outliers), 'ro', clip_on=False)

    kwargs = dict(y=0.95, x=0.05, ha='left', va='top')
    axes[0].set_title('Percentile-based Outliers', **kwargs)
    axes[1].set_title('MAD-based Outliers', **kwargs)
    fig.suptitle('Comparing Outlier Tests with n={}'.format(len(x)), size=14)

# find outliers for each assignment (store the index of the rows)
outliers_bool_ca1 = mad_based_outlier(result.ix[:, 'CA1'])
outliers_ca1_index = [i for i in range(len(outliers_bool_ca1)) if outliers_bool_ca1[i]==True]

outliers_bool_ca2 = mad_based_outlier(result.ix[:, 'CA2 Total'])
outliers_ca2_index = [i for i in range(len(outliers_bool_ca2)) if outliers_bool_ca2[i]==True]

outliers_bool_ca3 = mad_based_outlier(result.ix[:, 'CA3 Total'])
outliers_ca3_index = [i for i in range(len(outliers_bool_ca3)) if outliers_bool_ca3[i]==True]

outliers_bool_ca4 = mad_based_outlier(result.ix[:, 'CA4'])
outliers_ca4_index = [i for i in range(len(outliers_bool_ca4)) if outliers_bool_ca4[i]==True]

outliers_bool_final = mad_based_outlier(result.ix[:, 'Exam Total'])
outliers_final_index = [i for i in range(len(outliers_bool_final)) if outliers_bool_final[i]==True]

# split input and output
x = result.ix[:, ['Factor1', 'Factor2', 'CA1']]
y = result.ix[:, 'Final Total']

# convert categorical data to labels then convert labels to binary variables(one-hot encoding) to use regression

# lbl_enc = LabelEncoder()
# lbl_enc.fit(x['Factor1'])
# x_year = lbl_enc.transform(x['Factor1'])
# lbl_enc.fit(x['Factor2'])
# x_nat = lbl_enc.transform(x['Factor2'])
# ohe = OneHotEncoder()
# ohe.fit()

_x,_,_ = one_hot_dataframe(x, ['Factor1', 'Factor2'], replace=True)
# scaled_features = ['CA1', 'CA2 Total', 'CA3 Total', 'CA4']
scaled_features = ['CA1']


def candidate_families():
    candidates = []

    # Random Forest Regressor
    rfr = {'name': 'Random Forest Regressor',
           'estimator': RandomForestRegressor(n_jobs=-1, max_features='sqrt', n_estimators=50)}
    rfr_tuned_parameters = {'n_estimators': [300, 500, 800],
              #'max_depth': [5, 8, 15, 25, 30, None],
              #'min_samples_split': [1, 2, 5, 10, 15, 100],
              #'min_sample_leaf': [1, 2, 5, 10],
              'max_features': ['log2', 'sqrt', None]}
    candidates.append([rfr['name'], rfr, rfr_tuned_parameters])

    # Support vector regressor
    svr = {'name': 'support vector regression', 'estimator': SVR()}
    svr_tuned_parameters = {'C': [0.001, 0.01, 0.1, 1.10, 100, 1000, 10000], 'kernel': ['linear', 'rbf']}
    candidates.append([svr['name'], svr, svr_tuned_parameters])

    # Ridge regressor
    rir = {'name': 'Ridge regression', 'estimator': Ridge()}
    rir_tuned_parameters = {'alpha': [0.01, 0.1, 1.0, 10, 100], 'fit_intercept': [True, False], 'normalize': [True, False]}
    candidates.append([rir['name'], rir, rir_tuned_parameters])

    # Extra tree regressor
    etr = {'name': 'Extra tree regression', 'estimator': ExtraTreesRegressor(n_jobs=-1)}
    etr_tuned_parameters = {'n_estimators': [300, 500, 800],
                            'max_features': ['log2', 'sqrt', None]}
    candidates.append([etr['name'], etr, etr_tuned_parameters])

    # Lasso
    lasso = {'name': 'Lasso', 'estimator': Lasso()}
    lasso_tuned_parameters = {'alpha': [0.1, 1.0, 10],
                              'normalize': [True, False]}
    candidates.append([lasso['name'], lasso, lasso_tuned_parameters])

    # Elastic Net
    # enet = {'name': 'Elastic Net', 'estimator': ElasticNet(l1_ratio=0)}
    # enet_tuned_parameters = {'alpha': [0.1, 1.0, 10, 100],
    #                          'normalize': [True, False]}
    # candidates.append([enet['name'], enet, enet_tuned_parameters])

    return candidates

best_model = best_model(candidate_families(), scaling_feature(_x, meta_frame, *scaled_features), y)
print(best_model)

# cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# for model in candidate_families():
#     estimator = best_config(model[1], model[2], scaling_feature(_x, meta_frame, *scaled_features), y)
#
#     plot_learning_curve(estimator[2], "Learning Curve", scaling_feature(_x, meta_frame, *scaled_features), y)
#
#     print(cv_score_gen(estimator[2], scaling_feature(_x, meta_frame, *scaled_features), y, 4))


plt.show()
#print(cv_score_gen(best_model, _x, y, 4))


# TODO predict final results based on CA1 scores, CA2 scores,... according to time
def predict_final_scores_chronologically(*features):
    return None

