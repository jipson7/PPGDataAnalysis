import data
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd


def print_stats(trial_id, dl):
    wrist_device, _, true_device = dl.load_oxygen(trial_id, iid=False)
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
        if difference <= dl.threshold:
            true_reliable_count += 1

    actual_precent = (true_reliable_count / sample_count) * 100
    print("{}, or {:.1f}%, of labels were within {} of wrist sensor".format(true_reliable_count, actual_precent, dl.threshold))
    print("Positive Labels: " + str(true_reliable_count))


def visualize_classifier_results(training_ids, test_id, dl):

    X_train, y_train = dl.load(training_ids, iid=True)
    X_test, y_test = dl.load([test_id], iid=False)

    clf = xgb.XGBClassifier(
        learning_rate=0.015,
        n_estimators=250,
        max_depth=5,
        min_child_weight=5,
        gamma=0.0,
        subsample=0.8,
        colsample_bytree=0.9,
        objective='binary:logistic',
        nthread=data.N_JOBS,
        scale_pos_weight=3,
        reg_alpha=1e-6)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    wrist_oxygen, wrist_oxygen_clean, true_oxygen = dl.load_oxygen(test_id, y_pred=y_pred, iid=False)


    graph_df = pd.concat([wrist_oxygen, true_oxygen, wrist_oxygen_clean], axis=1, sort=True)

    assert(wrist_oxygen.shape == true_oxygen.shape)
    assert(graph_df.shape[0] == wrist_oxygen.shape[0])

    graph_df.plot.line(color=['red', 'blue', 'yellow'])
    plt.xlabel("Time")
    plt.ylabel("SpO2 (%)")
    plt.ylim()
    plt.savefig(data.GRAPH_CACHE + 'classifier-{}-{}.png'.format(test_id, str(dl)))


def print_all_stats():
    dl = data.DataLoader(window_size=100, threshold=1.0, algo_name='enhanced', features='comprehensive')
    for trial_id in data.list_trials():
        print("\nStats for trial: {}".format(trial_id))
        print_stats(trial_id, dl)


def visualize_all_classifier_results():
    trial_ids = [22, 23, 24, 29, 31, 32, 33, 36, 40, 43]

    dl = data.DataLoader(window_size=100, threshold=2.0, algo_name='enhanced', features='comprehensive')

    for trial_id in trial_ids:
        print("Trial {}".format(trial_id))
        training_ids = trial_ids.copy()
        training_ids.remove(trial_id)

        visualize_classifier_results(training_ids, trial_id, dl)

if __name__ == '__main__':
    print_all_stats()