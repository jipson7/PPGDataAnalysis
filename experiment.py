import data
import pickle
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from keras.wrappers.scikit_learn import KerasClassifier

# Used to suppress Fscore ill defined
warnings.filterwarnings('ignore')


MODEL_CACHE = './data-cache/models/'


def analyze_classifier(X, y, clf_og, params=None, n_jobs=-1):
    # Note the stratified K fold used by GridSearchCV does NOT shuffle.
    clf = GridSearchCV(clf_og, param_grid=params, scoring=['accuracy', 'f1_weighted'],
                       cv=5, verbose=1, refit='accuracy', n_jobs=n_jobs,
                       return_train_score=False, iid=False)
    clf.fit(X, y)
    results = clf.cv_results_
    clf_name = type(clf_og).__name__
    print(clf_name + " results:")
    for param, accuracy, f1 in zip(results['params'], results['mean_test_accuracy'], results['mean_test_f1_weighted']):
        print("Accuracy: {:.3f}, F1-Weighted: {:.3f}, Params: {}".format(accuracy, f1, param))
    print("\nBest Accuracy {}, Params: {}".format(clf.best_score_, clf.best_params_))
    estimator = clf.best_estimator_
    pickle_path = MODEL_CACHE + clf_name
    pickle.dump(estimator, open(pickle_path, "wb"))


def run_random_forest(X, y):
    print("\nRunning Random Forest")
    parameters = {
        'n_estimators': [x for x in range(20, 100, 10)],
        'criterion': ['gini']
    }
    analyze_classifier(X, y, RandomForestClassifier(), params=parameters)


def run_logistic_regression(X, y):
    print("\nRunning Logistic Regression")
    parameters = {
        'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
        'penalty': ('l1', 'l2')
    }
    analyze_classifier(X, y, LogisticRegression(), params=parameters)


def run_svc(X, y):
    print("\nRunning Support Vector Classification")
    parameters = [{
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'kernel': ['rbf', 'sigmoid']
    }, {
        'kernel': ['poly']
    }]
    analyze_classifier(X, y, SVC(), params=parameters)


def run_nn(X, y):
    print("\nRunning NN")
    # TODO Retune NN. Maybe run it on a beefier server?
    from neural_network import get_model_generator
    parameters = {
        'hidden_layers': [1, 2, 3, 4],
        'hidden_layer_size': [4, 8, 12],
        'optimizer': ['rmsprop', 'adam'],
        'init': ['glorot_uniform', 'normal', 'uniform'],
        'epochs': [1],
        'batch_size': [32]
    } # Maybe add loss?
    # parameters = {
    #     'epochs': [1, 2 ],
    #     'batch_size': [100, 250, 500, 1000]
    # }
    model_gen = get_model_generator(X.shape[1])
    clf = KerasClassifier(build_fn=model_gen, verbose=0)
    analyze_classifier(X, y, clf, params=parameters, n_jobs=1)


def run():
    data.list_trials()
    trial_id = 15
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
