import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor,export_graphviz
# import model_selection_util as ms
# import feature_transformer as ft

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from model_selection_util import best_config,best_model,cv_score_gen,plot_learning_curve
from feature_transformer import one_hot_dataframe, scaling_feature, encode_final_score, add_polynomial_features, \
                                remove_low_variance_features, feature_importance, extract_test_set
from model_generation import regression_families, classification_families, visualize_tree

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import ShuffleSplit

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
meta_data_1 = {'Max': [100, 40, 10, 50, 20, 30, 20, 70, 100, 100, 25, 25, 25, 25, 100, 100],
             'Weightage': [4, 9.6, 2.4, 12, 3.4, 5.2, 3.4, 12, 12, 40, 15, 15, 15, 15, 60, 100.00]}
meta_data_2 = {'Max': [4],
               'Weightage': [1]}
meta_frame_1 = pd.DataFrame(meta_data_1, columns=['Max', 'Weightage'], index=list(result.columns.values)[3:])
meta_frame_2 = pd.DataFrame(meta_data_2, columns=['Max', 'Weightage'], index=[list(result.columns.values)[1]])
meta_frame = pd.concat([meta_frame_1, meta_frame_2], axis=0)


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
    axes[1].set_title('Modified z score-based Outliers', **kwargs)
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

detect_outliers(result.ix[:, 'CA4'])
plt.show()

# split input and output
x = result.ix[:, ['Factor1', 'Factor2', 'CA1', 'CA2-1', 'CA2-2', 'CA3-1', 'CA3-2', 'CA3-3', 'CA4']]
y = result.ix[:, 'Exam Total'] # multi-variate problem, only Linear, Lasso, Ridge and random forest support


# convert categorical data to labels then convert labels to binary variables(one-hot encoding) to use regression
_x,_,_ = one_hot_dataframe(x, ['Factor1', 'Factor2'], replace=True)
# scaled_features = ['CA1', 'CA2 Total', 'CA3 Total', 'CA4']
scaled_features = ['Factor1', 'CA1', 'CA2-1', 'CA2-2', 'CA3-1', 'CA3-2', 'CA3-3', 'CA4'] # cannot scale Factor2
# poly_features = ['CA1', 'CA2 Total', 'CA3 Total', 'CA4']
poly_features = scaled_features

# processed_df = add_polynomial_features(scaling_feature(_x, meta_frame, *scaled_features), *poly_features, degree=3)
processed_df = scaling_feature(_x, meta_frame, *scaled_features)
dt = best_config({'name': 'Decision Tree Regressor', 'estimator': DecisionTreeRegressor()},
                 {'min_samples_split': np.arange(2, 10), 'min_samples_leaf': np.arange(2, 10), 'max_depth': np.arange(3, 10)}, processed_df, y)
print(cv_score_gen(dt[2], processed_df, y, 3))
visualize_tree(dt[2], list(processed_df.columns.values))

#feature_importance(processed_df, y, transform=False)

# test_input, test_output = extract_test_set(processed_df, y, [17, 38, 34, 45, 10, 83])
# processed_df = scaling_feature(_x, meta_frame, *scaled_features)

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# for model in regression_families():
#     estimator = best_config(model[1], model[2], processed_df, y)
#
#     plot_learning_curve(estimator[2], "Learning Curve", processed_df, y)
#
#     cv_score_gen(estimator[2], processed_df, y, 4)
    # estimator[2].fit(processed_df, y)
    # print("R2 score: %0.3f" % estimator[2].score(test_input, test_output))
    # print("                  predicted value\ttrue value ")
    # for index, row in test_input.iterrows():
    #     print("student%3d\t\t%0.3f\t\t%0.3f" % (index + 1, estimator[2].predict(row), test_output.ix[index]))
    #
    #plt.show()




# best_model = best_model(classification_families(), processed_df, y)

# print(best_model)




# TODO predict final results based on CA1 scores, CA2 scores,... according to time
def predict_final_scores_chronologically(*features):
    return None

