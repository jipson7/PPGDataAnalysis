import datetime
import data
import experiment as ex
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from examine import print_stats


def visualize_algorithms(trial_id, algo_name='enhanced', threshold=1.0):
    devices = data.load_devices(trial_id, algo_name=algo_name)

    wrist_oxygen = devices[0][['oxygen']]
    true_oxygen = devices[1][['oxygen']]

    wrist_oxygen.columns = ['Wrist Oxygen']
    true_oxygen.columns = ['Fingertip Oxygen']

    graph_df = pd.concat([wrist_oxygen, true_oxygen], axis=1, sort=True)

    assert(wrist_oxygen.shape == true_oxygen.shape)
    assert(graph_df.shape[0] == wrist_oxygen.shape[0])

    graph_df.plot.line(color=['red', 'blue'])
    plt.xlabel("Time")
    plt.ylabel("SpO2 (%)")
    plt.show()

    print_stats(wrist_oxygen, true_oxygen, threshold)


def visualize_classifier(trial_id, algo_name, threshold):
    clf = pickle.load(open('data-cache/classifier.pickle', "rb"))
    devices = data.load_devices(trial_id, algo_name=algo_name)
    wrist = devices[0]
    truth = devices[1]
    fe = data.FeatureExtractor(window_size=100, threshold=threshold)
    X, y_true = ex.create_training_data([trial_id], fe, algo_name)
    y_pred = clf.predict(X)

    # Remove first 99 elements to align with label
    wrist_oxygen = wrist[['oxygen']][99:]
    true_oxygen = truth[['oxygen']][99:]
    y_pred = y_pred.reshape(wrist_oxygen.shape)

    wrist_oxygen_clean = wrist_oxygen.where(y_pred)


    wrist_oxygen.columns = ['Wrist Oxygen']
    wrist_oxygen_clean.columns = ['Wrist Oxygen Reliable']
    true_oxygen.columns = ['Fingertip Oxygen']

    graph_df = pd.concat([wrist_oxygen, true_oxygen, wrist_oxygen_clean], axis=1, sort=True)

    assert(wrist_oxygen.shape == true_oxygen.shape)
    assert(graph_df.shape[0] == wrist_oxygen.shape[0])

    graph_df.plot.line(color=['red', 'blue', 'black'])
    plt.xlabel("Time")
    plt.ylabel("SpO2 (%)")
    plt.ylim(ymin=56, ymax=105)
    plt.show()

    print_stats(wrist_oxygen_clean, true_oxygen, threshold)

    # Get MSE
    mse_wrist = rmse(wrist_oxygen, true_oxygen)
    mse_wrist_clean = rmse(wrist_oxygen_clean, true_oxygen)
    print("RMSE Before: {}, After: {}".format(mse_wrist, mse_wrist_clean))

    # Longest Nan Wait
    n = max_consecutive_nans(wrist_oxygen_clean.values.flatten())
    longest_window = datetime.timedelta(seconds=(n * 40) / 1000)
    print("Longest NaN window: {}".format(longest_window)) # In seconds


def rmse(d1, d2):
    d1 = d1.values.flatten()
    d2 = d2.values.flatten()
    return math.sqrt(np.nanmean((np.subtract(d2, d1)) ** 2))


def max_consecutive_nans(a):
    mask = np.concatenate(([False], np.isnan(a), [False]))
    if ~mask.any():
        return 0
    else:
        idx = np.nonzero(mask[1:] != mask[:-1])[0]
        return (idx[1::2] - idx[::2]).max()


if __name__ == '__main__':
    trial_id = 23

    visualize_algorithms(trial_id, algo_name='enhanced', threshold=1.0)

    visualize_classifier(trial_id, algo_name='enhanced', threshold=1.0)
