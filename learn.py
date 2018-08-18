from server import app
from models import Trial
import data_morphing as dm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os.path

from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(42)

data_cache = './data-cache/'


def list_trials():
    with app.app_context():
        trials = Trial.query.all()
        for trial in trials:
            print(trial)


def load_devices(trial_id):
    print("\nLoading trial " + str(trial_id))

    pickle_path = data_cache + str(trial_id)

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


def run_random_forest(X_train, y_train, X_test, y_test):
    print('\nRunning Random Forest Classifier')
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    analyze_results(y_test, y_predicted)


def run_fnn(X_train, y_train, X_test, y_test):
    input_dimension = X_train.shape[1]
    num_classes = len(set(y_train) | set(y_test))
    # create model
    model = Sequential()
    model.add(Dense(input_dimension, input_dim=input_dimension, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Test model
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    analyze_results(y_test, y_predicted)


if __name__ == '__main__':
    list_trials()
    default_trial = 16
    devices = load_devices(default_trial)

    X_train, y_train, X_test, y_test = \
        dm.Experiment.oxygen_classification(wrist=devices['wrist'],
                                            oxygen_device=devices['reflective'],
                                            round_to=1)

    # run_random_forest(X_train, y_train, X_test, y_test)
    run_fnn(X_train, y_train, X_test, y_test)
