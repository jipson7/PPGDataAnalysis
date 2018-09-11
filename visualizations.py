import data
import experiment as ex
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def visualize_algorithms(trial_id, algo_name='enhanced', threshold=1.0):
    devices = data.load_devices(trial_id, algo_name=algo_name)

    wrist_oxygen = devices[0][['oxygen']]
    true_oxygen = devices[1][['oxygen']]

    print("Length of Dataframe: " + str(data.get_df_length(wrist_oxygen)))

    wrist_oxygen.columns = ['Wrist Oxygen']
    true_oxygen.columns = ['Fingertip Oxygen']

    graph_df = pd.concat([wrist_oxygen, true_oxygen], axis=1, sort=True)

    assert(wrist_oxygen.shape == true_oxygen.shape)
    assert(graph_df.shape[0] == wrist_oxygen.shape[0])

    graph_df.plot.line(color=['red', 'blue'])
    plt.xlabel("Time")
    plt.ylabel("SpO2 (%)")
    plt.savefig('figs/algo{}.png'.format(time.time()))
    plt.show()

    print_stats(wrist_oxygen, true_oxygen, threshold)


def print_stats(wrist_device, true_device, threshold):

    wrist_oxygen = wrist_device.values.flatten()
    true_oxygen = true_device.values.flatten()

    sample_count = wrist_oxygen.shape[0]
    wrist_reliable_count = np.count_nonzero(~np.isnan(wrist_oxygen))

    print("Samples Collected: " + str(sample_count))

    algo_percent = (wrist_reliable_count / sample_count) * 100
    print("Algorithm marked {} samples, or {:.1f}%, as reliable".format(wrist_reliable_count, algo_percent))

    true_reliable_count = 0
    for o1, o2 in zip(wrist_oxygen, true_oxygen):
        difference = np.abs(np.subtract(o1, o2))
        if difference <= threshold:
            true_reliable_count += 1

    actual_precent = (true_reliable_count / sample_count) * 100
    print("{}, or {:.1f}%, of labels were within {} of wrist sensor".format(true_reliable_count, actual_precent, threshold))


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
    plt.ylim(ymin=56, ymax=100)
    plt.savefig('figs/clf{}.png'.format(time.time()))
    plt.show()

    print_stats(wrist_oxygen, true_oxygen, threshold)


if __name__ == '__main__':
    trial_id = 22

    # visualize_algorithms(trial_id, algo_name='maxim', threshold=1.0)

    visualize_classifier(trial_id, algo_name='enhanced', threshold=1.0)
