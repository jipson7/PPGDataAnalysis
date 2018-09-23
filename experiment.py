import math
import warnings

import matplotlib
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import data
from data import N_JOBS, CM_CACHE, EXPERIMENT_CACHE

warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)


def optimize_classifier(trial_ids, data_loader):
    X, y = data_loader.load(trial_ids)
    parameters = {
        'learning_rate': [0.1, 0.3, 0.5],
        'n_estimators': [100, 1000],
        # 'max_depth': [3, 6, 9]
    }
    scoring = 'precision_weighted'

    cv = StratifiedKFold(n_splits=3, shuffle=False)

    clf = GridSearchCV(xgb.XGBClassifier(), param_grid=parameters, scoring=scoring,
                       cv=cv, verbose=3, refit=False, n_jobs=N_JOBS,
                       return_train_score=False, iid=False)
    clf.fit(X, y)
    results = clf.cv_results_
    print("XGB Optimal Model Developed")
    for param, score in zip(results['params'], results['mean_test_' + scoring]):
        print("{}: {:.3f}, Params: {}".format(scoring, score, param))
    print("\nBest {} {}, Params: {}".format(scoring, clf.best_score_, clf.best_params_))


def rmse(d1, d2):
    d1 = d1.values.flatten()
    d2 = d2.values.flatten()
    return math.sqrt(np.nanmean((np.subtract(d2, d1)) ** 2))


def run_experiments(clf, trial_ids, data_loader):
    log_name = EXPERIMENT_CACHE + "log-{}.txt".format(data_loader)
    log = open(log_name, 'w')
    cms = []
    precisions = []
    precisions_weighted = []
    for trial_id in trial_ids:
        start_msg = "Running experiments on trial " + str(trial_id) + '\n'
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
        precisions.append(precision_score(y_test, y_pred))
        precisions_weighted.append(precision_score(y_test, y_pred, average='weighted'))

        # Create confusion matrix
        cms.append(confusion_matrix(y_test, y_pred))

        # Get Before and after oxygen values
        wrist_oxygen, pruned_oxygen, fingertip_oxygen = data_loader.load_oxygen(trial_id, y_pred)

        # Get RMSE
        rmse_before = rmse(wrist_oxygen, fingertip_oxygen)
        rmse_after = rmse(pruned_oxygen, fingertip_oxygen)
        log.write("RMSE Before: {}, After: {}\n".format(rmse_before, rmse_after))

    # Create average confusion matrix
    avg_cm = np.average(cms, axis=0)
    avg_cm = np.array(avg_cm).astype(int)
    data.plot_confusion_matrix(avg_cm)
    plt.savefig(CM_CACHE + 'cm-' + str(data_loader) + '.png')

    log.write("Average Precision: {}".format(np.average(precisions)))
    log.write("Average Weighted Precision: {}".format(np.average(precisions_weighted)))

    log.close()


def run():
    trial_ids = [22, 23, 24, 29, 31, 32, 33, 36, 40, 43]
    clf = xgb.XGBClassifier(n_jobs=N_JOBS)  # Tune in tune.py
    dl = data.DataLoader(window_size=100, threshold=2.0, algo_name='maxim', features='comprehensive')
    run_experiments(clf, trial_ids, dl)


if __name__ == '__main__':
    run()

