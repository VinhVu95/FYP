import csv
import math
from sklearn.externals import joblib


def export_prediction_result_to_csv(model_name, table):
    """
    export prediction results to csv files
    Args:
        model_name: name of trained model to collect test results
        table: test results
        result will be rounded to two decimal places
    Returns: None

    """
    with open('Results/' + model_name + '.csv', 'w', newline='') as csvfile:
        fieldnames = ['student number', 'predicted value', 'true value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for num, pred, tv in table:
            writer.writerow({'student number': num+1, 'predicted value': round(pred, 2), 'true value': round(tv, 2)})

    return None


def save_model(model_name, model):
    """

    Args:
        model_name: name of trained model to save to hard disk
        model: trained model

    Returns: None

    """
    try:
        with open('Results/' + model_name + '.pkl', 'wb') as fid:
            joblib.dump(model, fid)
    except:
        exit("Could not save trained model to hard disk")

    return None
