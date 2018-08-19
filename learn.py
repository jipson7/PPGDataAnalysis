from server import app
from models import Trial
import data_morphing as dm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os.path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)


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


def analyze_results(y_true, y_predicted):
    accuracy = accuracy_score(y_true, y_predicted)
    print("Accuracy: {}".format(accuracy))

    plot_confusion_matrix(y_true, y_predicted)
    plt.show()


def run_random_forest(devices):
    X_train, y_train, X_test, y_test = \
        dm.Experiment.oxygen_classification(input_device=devices['wrist'],
                                            oxygen_device=devices['reflective'],
                                            round_to=1)


    print('\nRunning Random Forest Classifier')
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    analyze_results(y_test, y_predicted)


def run_fnn(X_train, y_train, X_test, y_test):
    y = np.concatenate((y_train, y_test))
    encoder = LabelEncoder()
    encoder.fit(y)

    y_train_hot = np_utils.to_categorical(encoder.transform(y_train))

    y_test_hot = np_utils.to_categorical(encoder.transform(y_test))

    EPOCHS = 1
    BATCH_SIZE = 1
    input_dimension = X_train.shape[1]
    num_classes = len(y_train_hot[0])
    # create model
    model = Sequential()
    model.add(Dense(100, input_dim=input_dimension, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Test model
    model.fit(X_train, y_train_hot, epochs=EPOCHS, batch_size=BATCH_SIZE)
    results = model.evaluate(X_test, y_test_hot)
    print(model.metrics_names)
    print(results)
    # print(y_predicted_hot)
    # print("Accuracy: {}".format(accuracy_score(y_test_hot, y_predicted_hot)))
    # print("F1 Score: {}".format(f1_score(y_test_hot, y_predicted_hot, average='weighted')))


def combine_trial_data(trials):
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    for i, trial in enumerate(trials):
        devices = load_devices(trial)
        X_tr, y_tr, X_te, y_te = \
            dm.Experiment.oxygen_classification(input_device=devices['wrist'],
                                                oxygen_device=devices['reflective'],
                                                round_to=0)
        if i == 0:
            X_train = X_tr
            y_train = y_tr
            X_test = X_te
            y_test = y_te
        else:
            X_train = np.concatenate([X_train, X_tr])
            y_train = np.concatenate([y_train, y_tr])
            X_test = np.concatenate([X_test, X_te])
            y_test = np.concatenate([y_test, y_te])
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    list_trials()

    X_train, y_train, X_test, y_test = combine_trial_data([15, 16])

    # TODO combine multiple datasets here since they're already prepped

    run_fnn(X_train, y_train, X_test, y_test)
