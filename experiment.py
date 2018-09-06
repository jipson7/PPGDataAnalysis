import data
import pickle
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score
import xgboost as xgb
import pandas as pd
import warnings

warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)


def pickle_features(clf, X):
    importances = clf.feature_importances_
    idx = np.flatnonzero(importances)
    features = X.columns[idx].tolist()
    print(features)
    pickle.dump(features, open('data-cache/features.pickle', "wb"))


def validate_classifier(clf, X, y):
    print("Valdiating classifier")
    y_pred = clf.predict(X)
    labels = sorted(np.unique(y_pred))
    cm = confusion_matrix(y, y_pred)
    print("Precision: " + str(precision_score(y, y_pred, average='weighted')))
    data.plot_confusion_matrix(cm, classes=labels, title="Confusion Matrix")
    plt.show()


def create_optimized_classifier(X, y, parameters):
    scoring = 'precision_weighted'

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    clf = GridSearchCV(xgb.XGBClassifier(), param_grid=parameters, scoring=['accuracy', scoring],
                       cv=cv, verbose=1, refit=scoring, n_jobs=-1,
                       return_train_score=False, iid=False)
    clf.fit(X, y)
    results = clf.cv_results_
    print("XGB Optimal Model Developed")
    for param, accuracy, score in zip(results['params'], results['mean_test_accuracy'], results['mean_test_' + scoring]):
        print("Accuracy: {:.3f}, {}: {:.3f}, Params: {}".format(accuracy, scoring, score, param))
    print("\nBest {} {}, Params: {}".format(scoring, clf.best_score_, clf.best_params_))
    return clf.best_estimator_


def create_training_data(trial_ids, feature_extractor, algo_name):
    X_s = []
    y_s = []
    for trial_id in trial_ids:
        devices = data.load_devices(trial_id, algo_name)
        X, y = feature_extractor.extract_features(devices)
        X_s.append(X)
        y_s.append(y)
    X = pd.concat(X_s, sort=True)
    y = pd.concat(y_s)
    print("Training Data Created")
    print("X: {}, y: {}".format(X.shape, y.shape))
    return X, y


if __name__ == '__main__':
    trial_ids = data.list_trials()

    ALGO_NAME = 'enhanced'

    fe = data.FeatureExtractor(window_size=100, threshold=3.0, from_pickle=False)
    training_trials = [20, 18, 13]
    X_train, y_train = create_training_data(training_trials, fe, algo_name=ALGO_NAME)

    # parameters = {
    #     'booster': ['gbtree', 'dart', 'gblinear'],
    #     'learning_rate': [0, 0.25, 0.5, 0.75, 1],
    #     'n_estimators': [x for x in range(10, 100, 10)],
    #     'objective': ['binary:logistic', 'binary:logitraw', 'binary:hinge']
    # }
    parameters = {
        'booster': ['gbtree'],
        'learning_rate': [0.1, 0.5, 0.75],
        'n_estimators': [10, 100, 200],
        'objective': ['binary:logistic', 'binary:logitraw', 'binary:hinge']
    }
    parameters = {}

    clf = create_optimized_classifier(X_train, y_train, parameters)
    pickle_features(clf, X_train)
    pickle.dump(clf, open('data-cache/classifier.pickle', "wb"))

    print("Prepping Validation Data")
    testing_trials = [22]
    X_test, y_test = create_training_data(testing_trials, fe, algo_name=ALGO_NAME)

    validate_classifier(clf, X_test, y_test)