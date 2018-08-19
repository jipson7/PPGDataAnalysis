import numpy as np
from data import normalize_timestamps


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
    def oxygen_classification(input_device=None, oxygen_device=None, round_to=1):
        print("\nPrepping Oxygen Prediction dataset...")
        if input_device is None or oxygen_device is None:
            raise ValueError("None valued device passed to data prep method")
        df_input_norm, df_oxygen_norm = normalize_timestamps(input_device, oxygen_device)

        input_columns = ['red', 'ir']
        if {'gyro', 'accel'}.issubset(df_input_norm.columns):
            input_columns = ['red', 'ir', 'gyro', 'accel']

        X = df_input_norm[input_columns].values
        #X = normalize(X, axis=0)
        y = df_oxygen_norm[['oxygen']].values

        y = y.reshape((X.shape[0],))

        # Round to nearest decimal place
        y = [round(i, round_to) for i in y]

        # Stringify label
        y = [str(i) for i in y]

        # Numpy encode
        y = np.array(y)

        print_label_counts(y)

        X_train, y_train, X_test, y_test = split_training_data(X, y)

        X_train, y_train = windowize_data(X_train, y_train)
        assert X_train.shape[0] == y_train.shape[0]
        X_test, y_test = windowize_data(X_test, y_test)
        assert X_test.shape[0] == y_test.shape[0]

        return X_train, y_train, X_test, y_test


