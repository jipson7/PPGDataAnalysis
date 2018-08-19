import datetime
import pandas as pd
from server import app
from models import Trial
import pickle
import os
import numpy as np

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


class FeatureExtractor:

    def __init__(self, window_size=100):
        self.window_size = window_size

    def window(self, iterable):
        i = iter(iterable)
        win = []
        for e in range(0, self.window_size):
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

            # Extract max for motion
            print(raw_sample.shape)
            exit(0)