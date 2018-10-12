import data
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
import trial_sets


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


def visualize_classifier_results(training_ids, test_id, dl, show_classifier=True):


    if show_classifier:
        X_train, y_train = dl.load(training_ids, iid=True)
        X_test, y_test = dl.load([test_id], iid=False)

        clf = xgb.XGBClassifier(
            learning_rate=0.1,
            n_estimators=101,
            max_depth=3,
            min_child_weight=3,
            gamma=0.3,
            subsample=0.9,
            colsample_bytree=0.6,
            scale_pos_weight=1,
            reg_alpha=0.01,
            objective='binary:logistic',
            nthread=data.N_JOBS,
            random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
    else:
        y_pred = None

    wrist_oxygen, wrist_oxygen_clean, true_oxygen = dl.load_oxygen(test_id, y_pred=y_pred, iid=False)

    if show_classifier:
        graph_df = pd.concat([wrist_oxygen, true_oxygen, wrist_oxygen_clean], axis=1, sort=True)
        colors = ['red', 'blue', 'yellow']
    else:
        graph_df = pd.concat([wrist_oxygen, true_oxygen], axis=1, sort=True)
        colors = ['red', 'blue']

    assert(wrist_oxygen.shape == true_oxygen.shape)
    assert(graph_df.shape[0] == wrist_oxygen.shape[0])

    graph_df.plot.line(color=colors)
    plt.xlabel("Time (Milliseconds)")
    plt.ylabel("SpO2 (%)")
    plt.ylim()
    if show_classifier:
        plt.savefig(data.GRAPH_CACHE + 'classifier-{}-{}.png'.format(test_id, str(dl)))
    else:
        plt.savefig(data.GRAPH_CACHE + 'algos-{}-{}.png'.format(test_id, str(dl)))


def print_all_stats():
    dl = data.DataLoader(window_size=100, threshold=1.0, algo_name='enhanced', features='comprehensive')
    for trial_id in trial_sets.top_ids:
        print("\nStats for trial: {}".format(trial_id))
        print_stats(trial_id, dl)


def visualize_all_classifier_results():

    trial_ids = trial_sets.top_ids

    dl = data.DataLoader(window_size=100, threshold=2.0, algo_name='enhanced', features='comprehensive')

    for trial_id in trial_ids:
        print("Trial {}".format(trial_id))
        training_ids = trial_ids.copy()
        training_ids.remove(trial_id)

        visualize_classifier_results(training_ids, trial_id, dl)


if __name__ == '__main__':
    # print_all_stats()
    visualize_all_classifier_results()