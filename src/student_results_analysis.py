import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from classes.predictor import Regressor
from process.model_selection_util import best_config, cv_score_gen,plot_learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeRegressor

from classes.outlier_detector import OutlierDetector
from cluster_analysis import visualise_cluster3d
from model_generation import regression_families, visualize_tree
from process.feature_transformer import one_hot_dataframe, scaling_feature, add_polynomial_features, \
    feature_importance

"""Pre-process raw data"""
data = pd.read_csv("Student Results.csv")
"""drop empty row from table"""
result = data.dropna()
"""drop student did not attend final"""
result = result[result['F'] != 'ABS']
"""after deleting row with student absent in final, the index not shift back -> reindex, and then drop the column 'index'"""
result = result.reset_index().drop('index', axis=1)

"""convert data type of marks to int and float"""
result.ix[:, 0] = result.ix[:, 0].astype(int)
result.ix[:, 3:7] = result.ix[:, 3:7].astype(float)
result.ix[:, 10:19] = result.ix[:, 10:19].astype(float)

"""store meta data for the table"""
meta_data_1 = {'Max': [100, 40, 10, 50, 20, 30, 20, 70, 100, 100, 25, 25, 25, 25, 100, 100],
             'Weightage': [4, 9.6, 2.4, 12, 3.4, 5.2, 3.4, 12, 12, 40, 15, 15, 15, 15, 60, 100.00]}
meta_data_2 = {'Max': [4],
               'Weightage': [1]}
meta_frame_1 = pd.DataFrame(meta_data_1, columns=['Max', 'Weightage'], index=list(result.columns.values)[3:])
meta_frame_2 = pd.DataFrame(meta_data_2, columns=['Max', 'Weightage'], index=[list(result.columns.values)[1]])
meta_frame = pd.concat([meta_frame_1, meta_frame_2], axis=0)


"""Detect outlier"""
od = OutlierDetector()
"""find outliers for each assignment (store the index of the rows)"""
outliers_bool_ca1 = od.mad_based_outlier(result.ix[:, 'C1'])
outliers_ca1_index = [i for i in range(len(outliers_bool_ca1)) if outliers_bool_ca1[i]==True]

outliers_bool_ca2 = od.mad_based_outlier(result.ix[:, 'C2 Total'])
outliers_ca2_index = [i for i in range(len(outliers_bool_ca2)) if outliers_bool_ca2[i]==True]

outliers_bool_ca3 = od.mad_based_outlier(result.ix[:, 'C3 Total'])
outliers_ca3_index = [i for i in range(len(outliers_bool_ca3)) if outliers_bool_ca3[i]==True]

outliers_bool_ca4 = od.mad_based_outlier(result.ix[:, 'C4'])
outliers_ca4_index = [i for i in range(len(outliers_bool_ca4)) if outliers_bool_ca4[i]==True]

outliers_bool_final = od.mad_based_outlier(result.ix[:, 'E'])
outliers_final_index = [i for i in range(len(outliers_bool_final)) if outliers_bool_final[i]==True]

x = result.ix[:, ['Background1', 'Background2', 'C1', 'C2-1', 'C2-2', 'C3-1', 'C3-2', 'C3-3', 'C4']]
print("input attributes :" + str(x.columns.tolist()))
print("output target: E")


def detect_outlier_students():
    """
        Detect outliers for each CA
    """
    od.detect_outliers(result.ix[:, 'C1'])
    od.detect_outliers(result.ix[:, 'C2 Total'])
    od.detect_outliers(result.ix[:, 'C3 Total'])
    od.detect_outliers(result.ix[:, 'C4'])
    plt.show()
    return None


def prepare():
    """
        Prepare data for decision tree visualization and calculation of feature ranking
    """
    x = result.ix[:, ['Background1', 'Background2', 'C1', 'C2-1', 'C2-2', 'C3-1', 'C3-2', 'C3-3', 'C4']]
    y = result.ix[:, 'E']
    "convert categorical data to labels then convert labels to binary variables(one-hot encoding) to use regression"
    _x, _, _ = one_hot_dataframe(x, ['Background1', 'Background2'], replace=True)
    scaled_features = ['Background1', 'C1', 'C2-1', 'C2-2', 'C3-1', 'C3-2', 'C3-3', 'C4']
    prepared_df = scaling_feature(_x, meta_frame, *scaled_features)
    return prepared_df, y


def visualize():
    """
        Visualize decision tree
    """
    prepared_df, y = prepare()
    dt = best_config(Regressor('Decision Tree Regressor', DecisionTreeRegressor(),
                               {'min_samples_split': [3], 'min_samples_leaf': [3],
                                'max_depth': [5]}), prepared_df, y)
    #print(cv_score_gen(dt[2], processed_df, y, 3))
    visualize_tree(dt[2], list(prepared_df.columns.values))
    return None


def calculate_feature_ranking():
    """
        Feature importance
    """
    prepared_df, y = prepare()
    imp, transformed_processed = feature_importance(prepared_df, y, transform=True)
    return None


def cluster():
    """
        Visualize clustering(grouping) of student data
    """
    prepared_df, y = prepare()
    num_clusters = [3,8,10]
    visualise_cluster3d(prepared_df, label=None, use_pca=True, *num_clusters)


def simulation(plot_learn_curve=False, add_poly=False, repeat=1):
    """
        Simulation function run the learning process n times and take average
        of cv score and prediction variance
    """
    models = regression_families()
    x = result.ix[:, ['C2-1', 'C2-2', 'C3-1', 'C3-2', 'C3-3']]
    y = result.ix[:, 'E']  # multi-variate problem, only Linear, Lasso, Ridge and random forest support

    "convert categorical data to labels then convert labels to binary variables(one-hot encoding) to use regression"
    # _x,_,_ = one_hot_dataframe(x, ['Background1', 'Background2'], replace=True)
    scaled_features = ['C2-1', 'C2-2', 'C3-1', 'C3-2', 'C3-3']  # cannot scale Factor2
    poly_features = scaled_features

    processed_df = scaling_feature(x, meta_frame, *scaled_features)
    if add_poly==True:
        kept_model_index = [0, 2, 3]
        processed_df = add_polynomial_features(scaling_feature(x, meta_frame, *scaled_features), *poly_features, degree=3)
        models = [i for j, i in enumerate(models) if j in kept_model_index]

    # test_input, test_output = extract_test_set(processed_df, y, [17, 38, 34, 45, 10, 83])
    # processed_df = scaling_feature(_x, meta_frame, *scaled_features)

    cv = ShuffleSplit(n_splits=7, test_size=0.2, random_state=0)
    pred_var = []
    val_score = []
    plot = [False] * len(models)
    if plot_learn_curve==True:
        plot = [True] * len(models)

    for i in range(repeat):
        max_prediction_variance = list()
        cv_score = list()
        for index, model in enumerate(models):
            estimator = best_config(model, processed_df, y)

            if plot[index] == True:
                plot_learning_curve(estimator[2], "Learning Curve", processed_df, y, cv=cv)
                plot[index] = False
                plt.show()

            max_score,_ = cv_score_gen(estimator[2], model.name, processed_df, y, 4)
            max_prediction_variance.append(max_score)
            cv_score.append(estimator[1])
            "if use test set to evaluate prediction variability"
            # estimator[2].fit(processed_df, y)
            # print("R2 score: %0.3f" % estimator[2].score(test_input, test_output))
            # print("                  predicted value\ttrue value ")
            # for index, row in test_input.iterrows():
            #     print("student%3d\t\t%0.3f\t\t%0.3f" % (index + 1, estimator[2].predict(row), test_output.ix[index]))
            "end"

        pred_var.append(max_prediction_variance)
        val_score.append(cv_score)

    np_max_prediction_variance = np.array(pred_var)
    np_val_score = np.array(val_score)

    print('average cv score:' + str(np_val_score.mean(0)))
    print('average prediction variability:' + str(np_max_prediction_variance.mean(0)))

    return None

repeat = 1

'''
########################################################################################################################
##--------------------------------------------DEMO STEPS--------------------------------------------------------------##
########################################################################################################################

'''
"1. Visualize decision tree to observe most important features"
# visualize()

"2. Feature importance"
# calculate_feature_ranking()

"3. Train classes and observe learning curve --> Simulation"
# simulation(plot_learn_curve=True)

"4. Increase polynomial degree for under-fit classes"
# simulation(plot_learn_curve=True,add_poly=True)

"5. Cluster analysis"
# cluster()

