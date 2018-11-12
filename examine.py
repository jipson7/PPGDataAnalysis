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


def create_error_cdf():
    THRESHOLD = 2.0
    dl_enhanced = data.DataLoader(window_size=100, threshold=THRESHOLD, algo_name='enhanced', features='comprehensive')
    dl_maxim = data.DataLoader(window_size=100, threshold=THRESHOLD, algo_name='maxim', features='comprehensive')

    maxim_errors = []
    enhanced_errors = []

    for trial_id in trial_sets.top_ids:
        wrist_enhanced, _, fingertip_enhanced = dl_enhanced.load_oxygen(trial_id, iid=False)
        wrist_maxim, _, fingertip_maxim = dl_maxim.load_oxygen(trial_id)

        wrist_maxim = wrist_maxim.values.flatten()
        wrist_enhanced = wrist_enhanced.values.flatten()
        fingertip_maxim = fingertip_maxim.values.flatten()
        fingertip_enhanced = fingertip_enhanced.values.flatten()

        for oM, oE, oMF, oEF in zip(wrist_maxim, wrist_enhanced, fingertip_maxim, fingertip_enhanced):
            maxim_errors.append(np.abs(np.subtract(oM, oMF)))
            enhanced_errors.append(np.abs(np.subtract(oE, oMF)))

    maxim_errors = np.array(maxim_errors)
    enhanced_errors = np.array(enhanced_errors)
    maxim_errors = maxim_errors[~np.isnan(maxim_errors)]
    enhanced_errors = enhanced_errors[~np.isnan(enhanced_errors)]
    rmses = [maxim_errors, enhanced_errors]

    plt.figure()

    for e in rmses:
        sorted_data = np.sort(e)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        plt.plot(sorted_data, yvals)

    plt.legend(['Baseline', 'Enhanced'])
    plt.ylim(0.0, 1.0)
    # plt.xlim(0.0, 10.0)
    plt.xlabel('MAE')

    plt.savefig(data.GRAPH_CACHE + 'cdf-error-all.png')


def create_fingertip_cdf():
    THRESHOLD = 2.0
    dl = data.DataLoader(window_size=100, threshold=THRESHOLD, algo_name='enhanced', features='comprehensive')

    fingertip_error = []

    for trial_id in trial_sets.top_ids:
        wrist_oxygen, fingertip_oxygen, transitive_oxygen = dl.load_all_oxygen(trial_id)

        for oF, oT in zip(fingertip_oxygen, transitive_oxygen):
            fingertip_error.append(np.abs(np.subtract(oF, oT)))

    fingertip_error = np.array(fingertip_error)
    fingertip_error = fingertip_error[~np.isnan(fingertip_error)]

    plt.figure()

    sorted_data = np.sort(fingertip_error)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
    plt.plot(sorted_data, yvals)

    # plt.legend(['Baseline', 'Enhanced'])
    plt.ylim(0.0, 1.0)
    plt.xlabel('MAE')

    plt.savefig(data.GRAPH_CACHE + 'cdf-fingertip.png')


if __name__ == '__main__':
    # print_all_stats()
    # visualize_all_classifier_results()
    # create_error_cdf()
    create_fingertip_cdf()
