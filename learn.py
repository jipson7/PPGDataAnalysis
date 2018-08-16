from server import app
from models import Trial
import data_morphing as dm
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(42)


def list_trials():
    with app.app_context():
        trials = Trial.query.all()
        for trial in trials:
            print(trial)


def load_devices(trial_id):
    print("\nLoading trial " + str(trial_id))
    with app.app_context():
        trial = Trial.query.get(trial_id)
        trial.get_info()
        devices = {'wrist': trial.df_wrist,
                   'reflective': trial.df_reflective,
                   'transitive': trial.df_transitive}
        print("Trial load finished.")
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


if __name__ == '__main__':
    list_trials()
    default_trial = 16
    devices = load_devices(default_trial)

    X_train, y_train, X_test, y_test = \
        dm.Experiment.oxygen_classification(wrist=devices['wrist'],
                                            oxygen_device=devices['reflective'],
                                            round_to=1)

    run_random_forest(X_train, y_train, X_test, y_test)
