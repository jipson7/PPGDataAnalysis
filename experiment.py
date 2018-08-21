import data
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from keras.wrappers.scikit_learn import KerasClassifier


def analyze_classifier(X, y, clf, params={}):
    clf = GridSearchCV(clf,
                       param_grid=params, scoring=['accuracy', 'f1'],
                       cv=5, verbose=1, refit='accuracy')
    clf.fit(X, y)
    print("\nAccuracy {}, Params: {}".format(clf.best_score_, clf.best_params_))


def run_random_forest(X, y):
    print("\nRunning Random Forest")
    parameters = {
        'n_estimators': [x for x in range(20, 100, 10)],
        'criterion': ['gini']
    }
    analyze_classifier(X, y, RandomForestClassifier(), parameters)


def run_logistic_regression(X, y):
    print("\nRunning Logistic Regression")
    parameters = {
        'C': [0.0001, 0.001, 0.01, 0.1, 1],
        'penalty': ('l1', 'l2')
    }
    analyze_classifier(X, y, LogisticRegression(), parameters)


def run_svc(X, y):
    print("\nRunning Logistic Regression")
    parameters = {
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10],
        'kernel': ['rbf', 'sigmoid']
    }
    analyze_classifier(X, y, SVC(), parameters)


def run_nn(X, y):
    print("\nRunning NN")
    # TODO Retune NN. Maybe run it on a beefier server?
    from neural_network import get_model_generator
    # parameters = {
    #     'hidden_layers': [1, 2],
    #     'hidden_layer_size': [8, 12],
    #     'optimizer': ['rmsprop', 'adam'],
    #     'init': ['glorot_uniform', 'normal', 'uniform'],
    #     'epochs': [1],
    #     'batch_size': [10, 50]
    # } # Maybe add loss?
    parameters = {
        'epochs': [1, 2 ],
        'batch_size': [100, 250, 500, 1000]
    }
    model_gen = get_model_generator(X.shape[1])
    clf = KerasClassifier(build_fn=model_gen, verbose=0)
    analyze_classifier(X, y, clf, parameters)


def run():
    data.list_trials()
    trial_id = 16
    devices = data.load_devices(trial_id)

    print("Extracting Wrist Features")
    fe = data.FeatureExtractor(window_size=100)
    X = fe.extract_wrist_features(devices[0])

    print("Creating Reliability labels")
    y = fe.create_reliability_label(devices)
    data.print_label_counts(y)

    run_random_forest(X, y)

    run_logistic_regression(X, y)

    run_svc(X, y)

    run_nn(X, y)


if __name__ == '__main__':
    run()
