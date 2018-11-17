import math
import datetime
import matplotlib

import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, precision_score
import trial_sets
matplotlib.use('Agg')

from matplotlib2tikz import save as tikz_save

import matplotlib.pyplot as plt
import data
from data import CM_CACHE, EXPERIMENT_CACHE, GRAPH_CACHE, LTX_CACHE


class Experiment(object):

    iid = False

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

    def __init__(self, experiment_name, data_loader, training_ids, validation_ids=None):
        self.experiment_name = experiment_name
        self.dl = data_loader
        if validation_ids is None:  # Cross validation
            folds = []
            for training_id in training_ids:
                x = training_ids.copy()
                x.remove(training_id)
                folds.append((x, training_id))
        else:
            folds = [(training_ids, test_id) for test_id in validation_ids]
        self.folds = folds
        self.log_name = EXPERIMENT_CACHE + "log-{}-{}.txt".format(data_loader, experiment_name)
        self.run()

    def run(self):
        log = open(self.log_name, 'w')
        log.write("Running Experiment {}\n".format(self.experiment_name))
        cms = []
        precisions = []
        rmse_befores = []
        rmse_afters = []
        nans = []

        for training_ids, test_id in self.folds:
            msg = "\nRunning {} against {} for {}\n".format(training_ids, test_id, self.experiment_name)
            print(msg)
            log.write(msg)
            X_train, y_train = self.dl.load(training_ids)
            self.clf.fit(X_train, y_train)
            X_test, y_test = self.dl.load([test_id], iid=self.iid)
            y_pred = self.clf.predict(X_test)
            # Get Precision Scores
            precision_weighted = precision_score(y_test, y_pred)
            precisions.append(precision_weighted)
            log.write("Precision {0:.1%}\n".format(precision_weighted))

            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            cms.append(cm)
            log.write("CM {}\n".format(cm))

            # Get Before and after oxygen values
            wrist_oxygen, pruned_oxygen, fingertip_oxygen = self.dl.load_oxygen(test_id, y_pred, iid=self.iid)

            # Get RMSE
            rmse_before = rmse(wrist_oxygen, fingertip_oxygen)
            rmse_befores.append(rmse_before)
            rmse_after = rmse(pruned_oxygen, fingertip_oxygen)
            rmse_afters.append(rmse_after)
            log.write("RMSE Before: {0:.1f}%\n".format(rmse_before))
            log.write("RMSE After: {0:.1f}%\n".format(rmse_after))

            # Longest Nan Wait
            n = max_consecutive_nans(pruned_oxygen.values.flatten())
            if self.iid:
                n *= 100
            longest_window = datetime.timedelta(seconds=(n * 40) / 1000)
            log.write("Longest NaN window: {}\n".format(longest_window))
            nans.append(longest_window.total_seconds())

        avg_cm = np.average(cms, axis=0)
        avg_cm = np.array(avg_cm).astype(int)
        # stddev_cm = np.std(cms, axis=0)
        # stddev_cm = np.array(stddev_cm).astype(int)
        data.plot_confusion_matrix(avg_cm)
        plt.savefig(CM_CACHE + 'cm-' + str(dl) + '-' + self.experiment_name + '.pdf', bbox_inches='tight')

        plot_cdf(rmse_befores, dl.algo.upper() + " Algorithm RMSE")
        log.write("RMSE before: {}\n".format(rmse_befores))
        plt.savefig(GRAPH_CACHE + 'cdf-{}-{}-rmse-before.pdf'.format(self.experiment_name, self.dl))
        # tikz_save(LTX_CACHE + 'cdf-{}-{}-rmse-before.tex'.format(self.experiment_name, self.dl))

        plot_cdf(rmse_afters, "Pruned RMSE")
        log.write("RMSE After: {}\n".format(rmse_afters))
        plt.savefig(GRAPH_CACHE + 'cdf-{}-{}-rmse-after.pdf'.format(self.experiment_name, self.dl))
        # tikz_save(LTX_CACHE + 'cdf-{}-{}-rmse-after.tex'.format(self.experiment_name, self.dl))

        plot_cdf(nans, "Time Between Readings (Seconds)")
        log.write("TIme Between readings: {}\n".format(nans))
        plt.savefig(GRAPH_CACHE + 'cdf-{}-{}-readings.pdf'.format(self.experiment_name, self.dl))
        # tikz_save(LTX_CACHE + 'cdf-{}-{}-readings.tex'.format(self.experiment_name, self.dl))

        log.write("Median Precision {}\n".format(np.nanmedian(precisions)))
        log.write("Median RMSE before {}\n".format(np.nanmedian(rmse_befores)))
        log.write("Median RMSE after {}\n".format(np.nanmedian(rmse_afters)))
        log.write("Median Time between readings {} (seconds)\n".format(np.nanmedian(nans)))

        log.write("Mean Precision {}\n".format(np.nanmean(precisions)))
        log.write("Mean RMSE before {}\n".format(np.nanmean(rmse_befores)))
        log.write("Mean RMSE after {}\n".format(np.nanmean(rmse_afters)))
        log.write("Mean Time between readings {} (seconds)\n".format(np.nanmean(nans)))

        log.write("Std Dev Precision {}\n".format(np.nanstd(precisions)))
        log.write("Std Dev RMSE before {}\n".format(np.nanstd(rmse_befores)))
        log.write("Std Dev RMSE after {}\n".format(np.nanstd(rmse_afters)))
        log.write("Std Dev Time between readings {} (seconds)\n".format(np.nanstd(nans)))

        log.close()


def plot_cdf(data, title):
    plt.figure(figsize=(4, 2))
    sorted_data = np.sort(data)
    yvals=np.arange(len(sorted_data))/float(len(sorted_data)-1)
    plt.plot(sorted_data,yvals)
    plt.ylim(0.0, 1.0)
    plt.xlabel(title)
    plt.tight_layout()


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


if __name__ == '__main__':

    dl = data.DataLoader(window_size=100, threshold=2.0, algo_name='enhanced', features='comprehensive')
    Experiment('top', dl, trial_sets.top_ids)

    # Experiment('bottom', dl, trial_sets.bottom_ids)
    # Experiment('top-bottom', dl, trial_sets.top_ids, validation_ids=trial_sets.bottom_ids)
    # Experiment('light', dl, trial_sets.top_light_ids)
    # Experiment('dark', dl, trial_sets.top_dark_ids)
    # Experiment('light-dark', dl, trial_sets.top_light_ids, validation_ids=trial_sets.top_dark_ids)
    # Experiment('dark-light', dl, trial_sets.top_dark_ids, validation_ids=trial_sets.top_light_ids)

    # Experiment('motion-only', dl, trial_sets.top_ids)

    # Experiment('self', dl, trial_sets.top_ids, validation_ids=[23])
    # Experiment('self-calibrated', dl, trial_sets.top_ids + [13], validation_ids=[23])
    # Experiment('self-more-calibrated', dl, trial_sets.top_ids + [13, 20], validation_ids=[23])


