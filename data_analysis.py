import datetime
import pandas as pd
import numpy as np
from itertools import islice
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


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


def windowize_data(X, y):
    X_windowed = []
    labels = []
    for X_window, y_window in zip(windowized(X), windowized(y)):
        X_window = np.array(X_window)
        row = []
        for column in X_window.T:
            row.extend(column)
        X_windowed.append(row)
        labels.append(y_window[-1])
    return np.array(X_windowed), np.array(labels)


def windowized(seq, n=100):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def plot_confusion_matrix(y_true, y_pred):
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')


def print_label_counts(y):
    from collections import Counter
    x = Counter(y)
    for label, count in x.items():
        print("Label: {}, Count: {}".format(label, count))


def analyze_results(y_true, y_predicted):
    accuracy = accuracy_score(y_true, y_predicted)
    print("Accuracy: {}".format(accuracy))

    plot_confusion_matrix(y_true, y_predicted)
    plt.show()


def split_training_data(X, y, ratio=0.66):
    data_count = X.shape[0]
    train_size = int(data_count * ratio)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[:train_size]

    return X_train, y_train, X_test, y_test