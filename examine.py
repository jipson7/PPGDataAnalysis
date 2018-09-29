import data
import numpy as np


def print_stats(wrist_device, true_device, threshold):

    print("Length of Dataframe: " + str(data.get_df_length(wrist_device)))

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
    print("Positive Labels: " + str(true_reliable_count))


if __name__ == '__main__':

    THRESHOLD = 1.0

    dl = data.DataLoader(window_size=100, threshold=THRESHOLD, algo_name='maxim', features='comprehensive')
    for trial_id in data.list_trials():

        wrist_oxygen, _, fingertip_oxygen = dl.load_oxygen(trial_id)
        print_stats(wrist_oxygen, fingertip_oxygen, THRESHOLD)
