import datetime
import pandas as pd
from server import app
from models import Trial
import pickle
import os
import numpy as np
from scipy.stats.stats import pearsonr

DATA_CACHE = './data-cache/'

np.random.seed(42)


def list_trials():
    with app.app_context():
        trials = Trial.query.all()
        for trial in trials:
            print(trial)


def load_devices(trial_id):
    print("\nLoading trial " + str(trial_id))

    pickle_path = DATA_CACHE + str(trial_id)

    if os.path.isfile(pickle_path):
        return pickle.load(open(pickle_path, "rb"))
    else:
        with app.app_context():
            trial = Trial.query.get(trial_id)
            trial.get_info()
            devices = {'wrist': trial.df_wrist,
                       'reflective': trial.df_reflective,
                       'transitive': trial.df_transitive}
            print("Trial load finished.")
            pickle.dump(devices, open(pickle_path, "wb"))
            return devices


def normalize_timestamps(df1, df2):

    def get_common_endpoints(df1, df2):
        df_start = df1.index[0] if (df1.index[0] > df2.index[0]) else df2.index[0]
        df_end = df1.index[-1] if (df1.index[-1] < df2.index[-1]) else df2.index[-1]
        return df_start, df_end

    sample_range = datetime.timedelta(milliseconds=40)
    indices = []
    new_data1 = []
    new_data2 = []
    df_start, df_end = get_common_endpoints(df1, df2)
    sample_date = df_start
    while sample_date < df_end:
        data1 = df1.iloc[df1.index.get_loc(sample_date, method='nearest')].values
        data2 = df2.iloc[df2.index.get_loc(sample_date, method='nearest')].values
        indices.append(sample_date)
        new_data1.append(data1)
        new_data2.append(data2)
        sample_date += sample_range
    new_df1 = pd.DataFrame(data=new_data1, index=indices, columns=df1.columns.values)
    new_df2 = pd.DataFrame(data=new_data2, index=indices, columns=df2.columns.values)
    return new_df1, new_df2


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

            # Pearson Correlation
            p_corr = np.corrcoef(led_traces, rowvar=False)[0, 1]
            p_correlation = p_corr if not np.isnan(p_corr) else 0
            feature_row.append(p_correlation)

            X.append(feature_row)
        return np.array(X)

    def extract_label(self, device, label='oxygen'):
        labels = device[[label]].values
        y = []
        for w in self.window(labels):
            y.extend(w[-1])
        return np.array(y)

    def create_reliability_label(self, real, predicted, threshold=1.0):
        y = np.subtract(real, predicted)
        y = np.abs(y)
        less_than_threshold = np.vectorize(lambda x: x <= threshold)
        y = less_than_threshold(y)
        return y


