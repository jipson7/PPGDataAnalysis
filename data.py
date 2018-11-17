import datetime
import pandas as pd
import pickle
import warnings
import os
import pathlib
from server import app
from models import Trial
import matplotlib.pyplot as plt
import numpy as np
import itertools
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters, EfficientFCParameters, MinimalFCParameters
from tsfresh.feature_selection.relevance import calculate_relevance_table

np.random.seed(42)
N_JOBS = 20
SAMPLE_LIMIT = 18000
CACHE_ROOT = './local-cache/'
CM_CACHE = CACHE_ROOT + 'cms/'
DATA_CACHE = CACHE_ROOT + 'data/'
EXPERIMENT_CACHE = CACHE_ROOT + 'experiments/'
FEATURE_CACHE = CACHE_ROOT + 'features/'
GRAPH_CACHE = CACHE_ROOT + 'graphs/'
LTX_CACHE = CACHE_ROOT + 'ltx/'

warnings.filterwarnings('ignore')


caches = [CM_CACHE, DATA_CACHE, EXPERIMENT_CACHE, FEATURE_CACHE, GRAPH_CACHE, LTX_CACHE]

for cache in caches:
    pathlib.Path(cache).mkdir(parents=True, exist_ok=True)


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
        data = data[-SAMPLE_LIMIT:]
        indices = indices[-SAMPLE_LIMIT:]
        result.append(pd.DataFrame(data=data, index=indices, columns=df.columns.values))
    return result


def print_label_counts(y):
    from collections import Counter
    x = Counter(y)
    for label, count in x.items():
        print("Label: {}, Count: {}".format(label, count))


def plot_confusion_matrix(cm,
                          normalize=False,
                          stddev=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    classes = ['Unreliable', 'Reliable']
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
        txt = format(cm[i, j], fmt)
        if stddev is not None:
            txt += '(±' + str(stddev[i, j]) + ')'
        plt.text(j, i, txt,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def get_df_length(df):
    return df.index[-1] - df.index[0]


class DataLoader:

    def __init__(self, window_size=100,
                 threshold=1.0, algo_name='maxim',
                 features='comprehensive', feature_limit=None,
                 motion=True):
        self.window_size = window_size
        self.threshold = threshold
        self.algo = algo_name
        self.feature_type = features
        self.selected_features = None
        self.feature_limit = feature_limit
        self.motion = motion

    def load(self, trial_ids, iid=True):
        X_s = []
        y_s = []
        for trial_id in trial_ids:
            devices = self._load_devices(trial_id)
            X = self._extract_features(devices, trial_id)
            y = pd.Series(data=self._create_reliability_label(devices))
            X.sort_index(axis=1, inplace=True)
            if iid:
                idx_iid = y.iloc[::self.window_size].index.values
                X = X.loc[idx_iid]
                y = y.loc[idx_iid]
            X_s.append(X)
            y_s.append(y)
        X = pd.concat(X_s, sort=True)
        y = pd.concat(y_s)
        if self.selected_features is None:
            rel_table = calculate_relevance_table(X, y, n_jobs=N_JOBS)
            rel_table = rel_table.loc[rel_table['relevant'] == True]
            sorted_features = rel_table.sort_values(by='p_value')
            feature_names = sorted_features.index.tolist()
            if self.feature_limit is not None:
                feature_names = feature_names[:self.feature_limit]
                assert len(feature_names) == self.feature_limit
            X = X[feature_names]
            self.selected_features = feature_names
        else:
            X = X[self.selected_features]
        print("Data loaded for trials: " + ', '.join([str(x) for x in trial_ids]))
        print("X shape: {}, y shape: {}".format(X.shape, y.shape))
        print_label_counts(y)
        # print("Features used: ")
        # from pprint import pprint
        # pprint(self.selected_features)
        return X, y

    def load_oxygen(self, trial_id, y_pred=None, iid=True):
        devices = self._load_devices(trial_id)

        wrist_oxygen = pd.DataFrame(self._extract_label(devices[0]))
        fingertip_oxygen = pd.DataFrame(self._extract_label(devices[1]))

        if iid:
            wrist_oxygen = wrist_oxygen.iloc[::self.window_size]
            fingertip_oxygen = fingertip_oxygen.iloc[::self.window_size]

        if y_pred is not None:
            pruned_oxygen = wrist_oxygen.where(y_pred.reshape(wrist_oxygen.shape))
            pruned_oxygen.columns = ['Wrist Oxygen Reliable']
        else:
            pruned_oxygen = None

        # Rename Columns
        wrist_oxygen.columns = ['Wrist Oxygen']
        fingertip_oxygen.columns = ['Fingertip Oxygen']

        return wrist_oxygen, pruned_oxygen, fingertip_oxygen

    def load_all_oxygen(self, trial_id):
        devices = self._load_devices(trial_id)
        wrist_oxygen = pd.DataFrame(self._extract_label(devices[0])).values.flatten()
        fingertip_oxygen = pd.DataFrame(self._extract_label(devices[1])).values.flatten()
        transitive_oxygen = pd.DataFrame(self._extract_label(devices[2])).values.flatten()
        return wrist_oxygen, fingertip_oxygen, transitive_oxygen

    def _load_devices(self, trial_id):
        pickle_path = DATA_CACHE + "{}-{}.pickle".format(trial_id, self.algo)
        if os.path.isfile(pickle_path):
            return pickle.load(open(pickle_path, "rb"))
        print("\nLoading trial {} with {} algorithm".format(trial_id, self.algo))
        with app.app_context():
            trial = Trial.query.get(trial_id)
            device_list = [trial.df_wrist(algo_name=self.algo),
                           trial.df_reflective(algo_name=self.algo),
                           trial.df_transitive()]
            device_list = normalize_timestamps(device_list)
            pickle.dump(device_list, open(pickle_path, "wb"))
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

    def _extract_features(self, devices, trial_id):

        if self.motion == True:
            pickle_path = FEATURE_CACHE + 'X{}-{}-{}.pickle'.format(trial_id, self.window_size, self.feature_type)
        elif self.motion == 'only':
            pickle_path = FEATURE_CACHE + 'X{}-{}-{}-motion-only.pickle'.format(trial_id, self.window_size, self.feature_type)
        else:
            pickle_path = FEATURE_CACHE + 'X{}-{}-{}-no-motion.pickle'.format(trial_id, self.window_size, self.feature_type)
        if os.path.isfile(pickle_path):
            return pickle.load(open(pickle_path, "rb"))
        else:

            wrist_device = devices[0]
            if self.motion == True:
                input_columns = ['red', 'ir', 'gyro', 'accel']
            elif self.motion == 'only':
                input_columns = ['gyro', 'accel']
            else:
                input_columns = ['red', 'ir']
            X_raw = wrist_device[input_columns]

            X_windowed = self._windowize_tsfresh(X_raw)

            if self.feature_type == 'efficient':
                features = EfficientFCParameters()
            elif self.feature_type == 'comprehensive':
                features = ComprehensiveFCParameters()
            elif self.feature_type == 'minimal':
                features = MinimalFCParameters()
            else:
                raise RuntimeError("Invalid feature type")
            print("Extracting features for trial " + str(trial_id))
            X = extract_features(X_windowed, column_id='id',
                                 column_sort='time',
                                 n_jobs=N_JOBS,
                                 default_fc_parameters=features)
            impute(X)
            pickle.dump(X, open(pickle_path, "wb"))
            return X

    def __str__(self):
        return "type-{}-{}-{}-{}".format(self.threshold, self.window_size, self.algo, self.feature_type)
