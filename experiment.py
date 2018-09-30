import math
import datetime
import matplotlib
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, precision_score

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import data
from data import CM_CACHE, EXPERIMENT_CACHE, GRAPH_CACHE

class Experiment(object):

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
        log_name = EXPERIMENT_CACHE + "log-{}-{}.txt".format(data_loader, experiment_name)
        self.log = open(log_name, 'w')
        self.run()

    def run(self):
        self.log.write("Running Experiment {}\n".format(self.experiment_name))
        cms = []
        precisions = []
        rmse_befores = []
        rmse_afters = []
        nans = []

        for training_ids, test_id in self.folds:
            msg = "\nRunning {} against {} for {}\n".format(training_ids, test_id, self.experiment_name)
            print(msg)
            self.log.write(msg)
            X_train, y_train = self.dl.load(training_ids)
            self.clf.fit(X_train, y_train)
            X_test, y_test = self.dl.load([test_id])
            y_pred = self.clf.predict(X_test)
            # Get Precision Scores
            precision_weighted = precision_score(y_test, y_pred, average='weighted')
            precisions.append(precision_weighted)
            self.log.write("Precision Weighted {0:.1%}\n".format(precision_weighted))

            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            cms.append(cm)
            self.log.write("CM {}\n".format(cm))

            # Get Before and after oxygen values
            wrist_oxygen, pruned_oxygen, fingertip_oxygen = self.dl.load_oxygen(test_id, y_pred)

            # Get RMSE
            rmse_before = rmse(wrist_oxygen, fingertip_oxygen)
            rmse_befores.append(rmse_before)
            rmse_after = rmse(pruned_oxygen, fingertip_oxygen)
            rmse_afters.append(rmse_after)
            self.log.write("RMSE Before: {0:.1f}%\n".format(rmse_before))
            self.log.write("RMSE After: {0:.1f}%\n".format(rmse_after))

            # Longest Nan Wait
            n = max_consecutive_nans(pruned_oxygen.values.flatten())
            longest_window = datetime.timedelta(seconds=(n * 40 * 100) / 1000)
            self.log.write("Longest NaN window: {}\n".format(longest_window))
            nans.append(longest_window.total_seconds())
        # Create average confusion matrix
        avg_cm = np.average(cms, axis=0)
        avg_cm = np.array(avg_cm).astype(int)
        data.plot_confusion_matrix(avg_cm)
        plt.savefig(CM_CACHE + 'cm-' + str(dl) + '-' + self.experiment_name + '.png')

        plot_cdf(rmse_befores, "Maxim Algorithm RMSE")
        self.log.write("RMSE before: {}\n".format(rmse_befores))
        plt.savefig(GRAPH_CACHE + 'cdf-' + str(dl) + '-rmse-before.png')

        plot_cdf(rmse_afters, "Pruned RMSE")
        self.log.write("RMSE After: {}\n".format(rmse_afters))
        plt.savefig(GRAPH_CACHE + 'cdf-' + str(dl) + '-rmse-after.png')

        plot_cdf(nans, "Time Between Readings (Seconds)")
        self.log.write("TIme Between readings: {}\n".format(nans))
        plt.savefig(GRAPH_CACHE + 'cdf-' + str(dl) + '-readings.png')

        self.log.write("Median Precision {}\n".format(np.nanmedian(precisions)))
        self.log.write("Median RMSE before {}\n".format(np.nanmedian(rmse_befores)))
        self.log.write("Median RMSE after {}\n".format(np.nanmedian(rmse_afters)))
        self.log.write("Median Time between readings {} (seconds)\n".format(np.nanmedian(nans)))

        self.log.close()


def plot_cdf(data, title):
    plt.figure()
    sorted_data = np.sort(data)
    yvals=np.arange(len(sorted_data))/float(len(sorted_data)-1)
    plt.plot(sorted_data,yvals)
    plt.ylim(0.0, 1.0)
    plt.xlabel(title)


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

    dark_ids = [43, 24, 40, 33, 36]  # Dark Skin
    light_ids = [22, 23, 29, 31, 32]  # Light skin
    all_ids = [22, 23, 24, 29, 31, 32, 33, 36, 40, 43]  # All 10

    dl = data.DataLoader(window_size=100, threshold=2.0, algo_name='maxim', features='comprehensive')

    Experiment('all', dl, all_ids)
    Experiment('light-dark', dl, light_ids, validation_ids=dark_ids)
    Experiment('dark-light', dl, dark_ids, validation_ids=light_ids)

