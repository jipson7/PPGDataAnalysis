import datetime
import pandas as pd
import numpy as np


def get_common_endpoints(df1, df2):
    df_start = df1.index[0] if (df1.index[0] > df2.index[0]) else df2.index[0]
    df_end = df1.index[-1] if (df1.index[-1] < df2.index[-1]) else df2.index[-1]
    return df_start, df_end


def normalize_timestamps(df1, df2):
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


def window(iterable, size=100):
    i = iter(iterable)
    win = []
    for e in range(0, size):
        win.append(next(i))
    yield win
    for e in i:
        win = win[1:] + [e]
        yield win


def windowize_data(X, y, size=100):
    # Flatten columns wise
    X_windowed = [np.array(i).flatten(order='F') for i in window(X)]
    labels = [i[-1] for i in window(y, size)]
    return np.array(X_windowed), np.array(labels)


def print_label_counts(y):
    from collections import Counter
    x = Counter(y)
    for label, count in x.items():
        print("Label: {}, Count: {}".format(label, count))


def split_training_data(X, y, ratio=0.66):
    data_count = X.shape[0]
    train_size = int(data_count * ratio)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    return X_train, y_train, X_test, y_test


class Experiment(object):

    @staticmethod
    def oxygen_classification(wrist=None, oxygen_device=None, round_to=1):
        print("\nPrepping Oxygen Prediction dataset...")
        if wrist is None or oxygen_device is None:
            raise ValueError("None value passed to data prep method")
        df_wrist_norm, df_transitive_norm = normalize_timestamps(wrist, oxygen_device)
        X = df_wrist_norm[['red', 'ir']].values
        y = df_transitive_norm[['oxygen']].values

        y = y.reshape((X.shape[0],))

        # Round to nearest decimal place
        y = [round(i, round_to) for i in y]

        # Stringify label
        y = np.array([str(i) for i in y])

        print_label_counts(y)

        X_train, y_train, X_test, y_test = split_training_data(X, y)

        X_train, y_train = windowize_data(X_train, y_train)
        X_test, y_test = windowize_data(X_test, y_test)

        return X_train, y_train, X_test, y_test
