import data
import numpy as np


def visualize_algorithms(trial_id, algo_name='enhanced', threshold=0.5):
    devices = data.load_devices(trial_id, algo_name=algo_name)
    wrist = devices[0]
    transitive = devices[2]

    wrist_oxygen = wrist[['oxygen']].values.flatten()
    true_oxygen = transitive[['oxygen']].values.flatten()

    sample_count = wrist_oxygen.shape[0]

    print("Analyzing {} algorithm with a threshold of {}".format(algo_name, threshold))

    assert(wrist_oxygen.shape == true_oxygen.shape)

    reliable_count = 0
    unreliable_count = 0

    for o1, o2 in zip(wrist_oxygen, true_oxygen):
        difference = np.abs(np.subtract(o1, o2))
        if difference <= threshold:
            reliable_count += 1
        else:
            unreliable_count += 1
    print("Reliable Labels: {}".format(reliable_count))
    print("Unreliable Labels: {}".format(unreliable_count))
    percent = (reliable_count / sample_count) * 100
    print("{:.1f}% of labels are within {}".format(percent, threshold))



if __name__ == '__main__':
    trial_id = 18
    visualize_algorithms(trial_id, algo_name='enhanced')
