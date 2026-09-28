import numpy as np

def calculate_rmse(
        prediction,
        ground_truth
):
    squared_error = (prediction - ground_truth) ** 2
    mean_squared_error = np.mean(squared_error)
    rmse = np.sqrt(mean_squared_error)

    return rmse


def calculate_mae(
        prediction,
        ground_truth
):
    absolute_error = np.abs(prediction - ground_truth)
    mae = np.mean(absolute_error)

    return mae

def calculate_mape(
        prediction,
        ground_truth
):
    absolute_percentage_error = np.abs((prediction - ground_truth) / ground_truth)
    mape = np.mean(absolute_percentage_error)

    return mape

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
