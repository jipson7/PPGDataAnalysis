import math
import datetime
import matplotlib
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, precision_score

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import data
from data import CM_CACHE, EXPERIMENT_CACHE


def rmse(d1, d2):
    assert d1.shape == d2.shape
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


def run_experiments(clf, trial_ids, data_loader):
    log_name = EXPERIMENT_CACHE + "log-{}.txt".format(data_loader)
    log = open(log_name, 'w')
    cms = []
    precisions_weighted = []
    avg_rmse_before = []
    avg_rmse_after = []
    for trial_id in trial_ids:
        start_msg = "\n\nRunning experiments on trial " + str(trial_id) + "\n"
        log.write(start_msg)
        print(start_msg)

        # Prep leave 1 out data
        training_ids = trial_ids.copy()
        training_ids.remove(trial_id)
        X_train, y_train = data_loader.load(training_ids)
        clf.fit(X_train, y_train)
        X_test, y_test = data_loader.load([trial_id])
        y_pred = clf.predict(X_test)

        # Get Precision Scores
        precision_weighted = precision_score(y_test, y_pred, average='weighted')
        log.write("Precision Weighted {0:.1%}\n".format(precision_weighted))
        print("Precision Weighted {0:.1%}\n".format(precision_weighted))
        precisions_weighted.append(precision_weighted)

        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        log.write("CM {}\n".format(cm))
        cms.append(cm)

        # Get Before and after oxygen values
        wrist_oxygen, pruned_oxygen, fingertip_oxygen = data_loader.load_oxygen(trial_id, y_pred)

        # Get RMSE
        rmse_before = rmse(wrist_oxygen, fingertip_oxygen)
        avg_rmse_before.append(rmse_before)
        rmse_after = rmse(pruned_oxygen, fingertip_oxygen)
        avg_rmse_after.append(rmse_after)
        rmse_result = ("RMSE Before: {0:.1f}%\n".format(rmse_before))
        print(rmse_result)
        log.write(rmse_result)
        rmse_result = ("RMSE After: {0:.1f}%\n".format(rmse_after))
        print(rmse_result)
        log.write(rmse_result)

        # Longest Nan Wait
        n = max_consecutive_nans(pruned_oxygen.values.flatten())
        longest_window = datetime.timedelta(seconds=(n * 40 * 100) / 1000)
        print("Longest NaN window: {}\n".format(longest_window))
        log.write("Longest NaN window: {}\n".format(longest_window))

    print('\nResults:\n')
    # Create average confusion matrix
    avg_cm = np.average(cms, axis=0)
    avg_cm = np.array(avg_cm).astype(int)
    data.plot_confusion_matrix(avg_cm)
    plt.savefig(CM_CACHE + 'cm-' + str(data_loader) + '.png')
    plt.show()

    log.write("Average Weighted Precision: {0:.1%}\n".format(np.average(precisions_weighted)))

    log.write("Average RMSE Before: {0:.1f}%, After: {0:.1f}%\n".format(np.nanmean(avg_rmse_before), np.nanmean(avg_rmse_after)))
    log.close()


def run():
    # trial_ids = [43, 24, 40, 33, 36]  # Dark Skin
    # trial_ids = [22, 23, 29, 31, 32, 20]  # Light skin
    trial_ids = [22, 23, 24, 29, 31, 32, 33, 36, 40, 43]  # All 10
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

    for t in [1.0]:
        dl = data.DataLoader(window_size=100, threshold=t, algo_name='maxim', features='comprehensive')
        run_experiments(clf, trial_ids, dl)


if __name__ == '__main__':
    run()

