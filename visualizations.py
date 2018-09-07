import data
import numpy as np


def visualize_algorithms(trial_id, algo_name='enhanced', threshold=2.0):
    devices = data.load_devices(trial_id, algo_name=algo_name)
    wrist = devices[0]
    transitive = devices[2]

    df_length = data.get_df_length(wrist)

    print("Length of Dataframe: " + str(df_length))

    wrist_oxygen = wrist[['oxygen']].values.flatten()
    true_oxygen = transitive[['oxygen']].values.flatten()

    sample_count = wrist_oxygen.shape[0]
    wrist_reliable_count = np.count_nonzero(~np.isnan(wrist_oxygen))

    print("Samples Collected: " + str(sample_count))

    algo_percent = (wrist_reliable_count / sample_count) * 100
    print("{} algorithm marked {} samples, or {:.1f}%, as reliable".format(algo_name, wrist_reliable_count, algo_percent))

    assert(wrist_oxygen.shape == true_oxygen.shape)

    true_reliable_count = 0
    for o1, o2 in zip(wrist_oxygen, true_oxygen):
        difference = np.abs(np.subtract(o1, o2))
        if difference <= threshold:
            true_reliable_count += 1

    actual_precent = (true_reliable_count / sample_count) * 100
    print("{}, or {:.1f}%, of labels were within {} of transitive sensor".format(true_reliable_count, actual_precent, threshold))


if __name__ == '__main__':
    trial_id = 21
    visualize_algorithms(trial_id, algo_name='enhanced')
