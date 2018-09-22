import datetime
import pandas as pd
import pickle
import os
from server import app
from models import Trial
import matplotlib.pyplot as plt
import numpy as np
import itertools
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters, EfficientFCParameters


np.random.seed(42)
N_JOBS = 20
CACHE_ROOT = './local-cache/'


def list_trials():
    with app.app_context():
        trials = Trial.query.all()
        for trial in trials:
            print(trial)
        return sorted([trial.id for trial in trials])


def normalize_timestamps(dataframes):
    """Returns the start and end of overlapping window
    between all N dataframes"""

    print("\nNormalizing Timestamps between {} devices".format(len(dataframes)))

    def get_common_endpoints(dfs):
        return max([x.index[0] for x in dfs]),\
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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def get_df_length(df):
    return df.index[-1] - df.index[0]


class DataLoader:

    def __init__(self, window_size=100, threshold=1.0, algo_name='maxim', features=None):
        self.window_size = window_size
        self.threshold = threshold
        self.algo = algo_name
        self.features = features

    def load(self, trial_ids):
        X_s = []
        y_s = []
        for trial_id in trial_ids:
            pickle_path = CACHE_ROOT + 'xy/trial' + str(trial_id) + str(self) + '.pickle'
            if os.path.isfile(pickle_path):
                Xy = pickle.load(open(pickle_path, "rb"))
                X = Xy[0]
                y = Xy[1]
            else:
                X, y = self._extract_features(trial_id)
                X.sort_index(axis=1, inplace=True)
                pickle.dump([X, y], open(pickle_path, "wb"))
            X_s.append(X)
            y_s.append(y)
        X = pd.concat(X_s, sort=True)
        y = pd.concat(y_s)
        print("Filtering features")
        X = select_features(X, y, n_jobs=N_JOBS)
        print("Training Data Created")
        print("X: {}, y: {}".format(X.shape, y.shape))
        return X, y

    def _load_devices(self, trial_id):
        print("\nLoading trial {} with {} algorithm".format(trial_id, self.algo))
        with app.app_context():
            trial = Trial.query.get(trial_id)
            device_list = [trial.df_wrist(algo_name=self.algo),
                           trial.df_reflective(algo_name=self.algo),
                           trial.df_transitive()]
            device_list = normalize_timestamps(device_list)
            print("Trial load finished.")
            return device_list

    def _window(self, iterable):
        i = iter(iterable)
        win = []
        for e in range(0, self.window_size):
            win.append(next(i))
        yield np.array(win)
        for e in i:
            win = win[1:] + [e]
            yield np.array(win)

    def _extract_label(self, device, label='oxygen'):
        labels = device[[label]].values
        y = []
        for w in self._window(labels):
            y.extend(w[-1])
        return np.array(y)

    """
    Threshold is defined as the largest distance (max) between any 2 non-null
    O2 Values measured across devices.
    """
    def _create_reliability_label(self, devices):
        print("Creating Reliability Labels")
        from itertools import combinations
        labels = []
        devices = [devices[0], devices[1]]  # Throw away transitive device
        for device in devices:
            labels.append(self._extract_label(device))
        errors = []
        for label1, label2 in combinations(labels, 2):
            try:
                diff = np.abs(np.subtract(label1, label2))
            except TypeError:
                raise RuntimeError("Unable to find algorithms. Remember to apply binary to trial")
            errors.append(diff)

        y = []
        for row_error in np.array(errors).T:
            if np.isnan(row_error).any() or row_error.max() > self.threshold:
                y.append(False)
            else:
                y.append(True)
        y = np.array(y)
        print_label_counts(y)
        return y

    def _windowize_tsfresh(self, X_raw):
        column_names = np.concatenate((['id', 'time'], X_raw.columns.values))

        X_windowed = []

        print("Windowizing data")
        for label_idx, window in enumerate(self._window(X_raw.itertuples())):
            for w in window:
                w = np.insert(w, 0, label_idx)
                X_windowed.append(w)

        return pd.DataFrame(X_windowed, columns=column_names)

    def _extract_features(self, trial_id):
        devices = self._load_devices(trial_id)
        wrist_device = devices[0]
        input_columns = ['red', 'ir', 'gyro', 'accel']
        X_raw = wrist_device[input_columns]

        X_windowed = self._windowize_tsfresh(X_raw)

        y = pd.Series(data=self._create_reliability_label(devices))

        if self.features is None:
            X = extract_features(X_windowed, column_id='id',
                                 column_sort='time',
                                 n_jobs=N_JOBS,
                                 default_fc_parameters=EfficientFCParameters())
        else:
            X = extract_features(X_windowed, column_id='id',
                                 column_sort='time',
                                 n_jobs=N_JOBS,
                                 kind_to_fc_parameters=self.features)

        impute(X)

        print("{} features extracted for trial {}".format(X.shape[1], trial_id))
        return X, y

    def __str__(self):
        return "type-{}-{}-{}".format(self.threshold, self.window_size, self.algo)
