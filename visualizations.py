import data
import experiment as ex
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def visualize_algorithms(trial_id, algo_name='enhanced', threshold=2.0):
    devices = data.load_devices(trial_id, algo_name=algo_name)
    wrist = devices[0]
    transitive = devices[2]

    df_length = data.get_df_length(wrist)

    print("Length of Dataframe: " + str(df_length))

    wrist_oxygen = wrist[['oxygen']]
    true_oxygen = transitive[['oxygen']]

    wrist_oxygen.columns = ['Wrist Oxygen']
    true_oxygen.columns = ['Fingertip Oxygen']

    graph_df = pd.concat([wrist_oxygen, true_oxygen], axis=1, sort=True)

    assert(wrist_oxygen.shape == true_oxygen.shape)
    assert(graph_df.shape[0] == wrist_oxygen.shape[0])

    graph_df.plot.line()
    plt.xlabel("Time")
    plt.ylabel("SpO2 (%)")
    plt.savefig('figs/algo{}.png'.format(time.time()))
    plt.show()

    wrist_oxygen = wrist_oxygen.values.flatten()
    true_oxygen = true_oxygen.values.flatten()

    sample_count = wrist_oxygen.shape[0]
    wrist_reliable_count = np.count_nonzero(~np.isnan(wrist_oxygen))

    print("Samples Collected: " + str(sample_count))

    algo_percent = (wrist_reliable_count / sample_count) * 100
    print("{} algorithm marked {} samples, or {:.1f}%, as reliable".format(algo_name, wrist_reliable_count, algo_percent))

    true_reliable_count = 0
    for o1, o2 in zip(wrist_oxygen, true_oxygen):
        difference = np.abs(np.subtract(o1, o2))
        if difference <= threshold:
            true_reliable_count += 1

    actual_precent = (true_reliable_count / sample_count) * 100
    print("{}, or {:.1f}%, of labels were within {} of transitive sensor".format(true_reliable_count, actual_precent, threshold))


def visualize_classifier(trial_id, algo_name, threshold):
    clf = pickle.load(open('data-cache/classifier.pickle', "rb"))
    devices = data.load_devices(trial_id, algo_name=algo_name)
    wrist = devices[0]
    transitive = devices[2]
    fe = data.FeatureExtractor(window_size=100, threshold=threshold, from_pickle=True)
    X, y_true = ex.create_training_data([trial_id], fe, algo_name)
    y_pred = clf.predict(X)


    wrist_oxygen = wrist[['oxygen']].where(y_pred)
    true_oxygen = transitive[['oxygen']]

    wrist_oxygen.columns = ['Wrist Oxygen']
    true_oxygen.columns = ['Fingertip Oxygen']



    graph_df = pd.concat([wrist_oxygen, true_oxygen], axis=1, sort=True)

    assert(wrist_oxygen.shape == true_oxygen.shape)
    assert(graph_df.shape[0] == wrist_oxygen.shape[0])

    graph_df.plot.line()
    plt.xlabel("Time")
    plt.ylabel("SpO2 (%)")
    plt.savefig('figs/clf{}.png'.format(time.time()))
    plt.show()



if __name__ == '__main__':
    trial_id = 22

    # visualize_algorithms(trial_id, algo_name='enhanced')

    visualize_classifier(trial_id, algo_name='enhanced', threshold=2.5)
