import datetime
import pandas as pd
from server import app
from models import Trial
import pickle
import os
import numpy as np

TRIAL_CACHE = './data-cache/trials/'

np.random.seed(42)


def list_trials():
    with app.app_context():
        trials = Trial.query.all()
        for trial in trials:
            print(trial)


def load_devices(trial_id):
    print("\nLoading trial " + str(trial_id))

    pickle_path = TRIAL_CACHE + str(trial_id)

    if os.path.isfile(pickle_path):
        return pickle.load(open(pickle_path, "rb"))
    else:
        with app.app_context():
            trial = Trial.query.get(trial_id)
            trial.get_info()
            device_list = \
                normalize_timestamps([trial.df_wrist,
                                      trial.df_reflective,
                                      trial.df_transitive])
            print("Trial load finished.")
            pickle.dump(device_list, open(pickle_path, "wb"))
            return device_list


def normalize_timestamps(dataframes):
    """Returns the start and end of overlapping window
    between all N dataframes"""

    print("\nNormalizing Timestamps between {} devices".format(len(dataframes)))

    def get_common_endpoints(dfs):
        return max([x.index[1] for x in dfs]),\
               min([x.index[-1] for x in dfs])

    sample_range = datetime.timedelta(milliseconds=40)
    indices = []
    new_data = [[] for _ in range(len(dataframes))]
    df_start, df_end = get_common_endpoints(dataframes)
    sample_date = df_start
    while sample_date < df_end:
        for i, df in enumerate(dataframes):
            data = df.iloc[df.index.get_loc(sample_date, method='nearest')].values
            new_data[i].append(data)
        indices.append(sample_date)
        sample_date += sample_range
    result = []
    for df, data in zip(dataframes, new_data):
        result.append(pd.DataFrame(data=data, index=indices, columns=df.columns.values))
    return result


def print_label_counts(y):
    from collections import Counter
    x = Counter(y)
    for label, count in x.items():
        print("Label: {}, Count: {}".format(label, count))


class FeatureExtractor:
    def __init__(self, window_size=100):
        self._window_size = window_size

    def window(self, iterable):
        i = iter(iterable)
        win = []
        for e in range(0, self._window_size):
            win.append(next(i))
        yield np.array(win)
        for e in i:
            win = win[1:] + [e]
            yield np.array(win)

    """
    Should normalize timestamps between 2 devices before
    passing to this function.
    """
    def extract_wrist_features(self, wrist_device):
        X = []
        input_columns = ['red', 'ir', 'gyro', 'accel']
        X_raw = wrist_device[input_columns].values

        for raw_sample in self.window(X_raw):
            feature_row = []
            led_traces = raw_sample[:, [0, 1]]
            motion_traces = raw_sample[:, [2, 3]]

            """Motion Features"""

            # Max
            feature_row.extend(motion_traces.max(axis=0))

            """LED Features"""

            # Mean
            feature_row.extend(led_traces.mean(axis=0))

            # StdDev
            feature_row.extend(led_traces.std(axis=0))

            # Max
            feature_row.extend(led_traces.max(axis=0))

            # Min
            feature_row.extend(led_traces.min(axis=0))

            # Pearson Correlation
            p_corr = np.corrcoef(led_traces, rowvar=False)[0, 1]
            p_correlation = p_corr if not np.isnan(p_corr) else 0
            feature_row.append(p_correlation)

            X.append(feature_row)
        return np.array(X)

    def _extract_label(self, device, label='oxygen'):
        labels = device[[label]].values
        y = []
        for w in self.window(labels):
            y.extend(w[-1])
        return np.array(y)

    """
    Threshold is defined as the largest distance (max) between any 2 non-null
    O2 Values measured across devices.
    """
    def create_reliability_label(self, devices, threshold=3.0):
        from itertools import combinations
        labels = []
        for device in devices:
            labels.append(self._extract_label(device))
        errors = []
        for label1, label2 in combinations(labels, 2):
            diff = np.abs(np.subtract(label1, label2))
            errors.append(diff)

        y = []
        for row_error in np.array(errors).T:
            if np.isnan(row_error).any() or row_error.max() > threshold:
                y.append(False)
            else:
                y.append(True)
        return np.array(y)


