import math
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


def run_experiments(clf, trial_ids, data_loader):
    log_name = EXPERIMENT_CACHE + "log-{}.txt".format(data_loader)
    log = open(log_name, 'w')
    cms = []
    precisions = []
    precisions_weighted = []
    avg_rmse_before = []
    avg_rmse_after = []
    for trial_id in trial_ids:
        start_msg = "\nRunning experiments on trial " + str(trial_id) + '\n'
        log.write(start_msg)
        print(start_msg)

        # Prep leave 1 out data
        print("Fitting classifier")
        training_ids = trial_ids.copy()
        training_ids.remove(trial_id)
        X_train, y_train = data_loader.load(training_ids)
        clf.fit(X_train, y_train)
        X_test, y_test = data_loader.load([trial_id])
        y_pred = clf.predict(X_test)

        print("Calculating Results")
        # Get Precision Scores
        precision = precision_score(y_test, y_pred)
        log.write("Precision {}\n".format(precision))
        precisions.append(precision)
        precision_weighted = precision_score(y_test, y_pred, average='weighted')
        log.write("Precision Weighted{}\n".format(precision_weighted))
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
        rmse_result = ("RMSE Before: {}, After: {}\n".format(rmse_before, rmse_after))
        print(rmse_result)
        log.write(rmse_result)

    print('\nResults:\n')
    # Create average confusion matrix
    avg_cm = np.average(cms, axis=0)
    avg_cm = np.array(avg_cm).astype(int)
    data.plot_confusion_matrix(avg_cm)
    plt.savefig(CM_CACHE + 'cm-' + str(data_loader) + '.png')

    log.write("Average Precision: {}\n".format(np.average(precisions)))
    log.write("Average Weighted Precision: {}\n".format(np.average(precisions_weighted)))

    log.write("Average RMSE Before: {}, After: {}\n".format(np.average(avg_rmse_before), np.average(avg_rmse_after)))
    log.close()


def run():
    trial_ids = [22, 23, 24, 29, 31, 32, 33, 36, 40, 43]
    clf = xgb.XGBClassifier(
        learning_rate=0.015,
        n_estimators=250,
        max_depth=4,
        min_child_weight=7,
        gamma=0.1,
        subsample=0.7,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=data.N_JOBS,
        scale_pos_weight=1,
        reg_alpha=1e-6)
    dl = data.DataLoader(window_size=100, threshold=3.0, algo_name='maxim', features='comprehensive')
    run_experiments(clf, trial_ids, dl)


if __name__ == '__main__':
    run()

