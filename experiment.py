import data
import pickle
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score
import xgboost as xgb
import warnings
from data import N_JOBS, CACHE_ROOT
from tsfresh.feature_extraction.settings import from_columns

warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)


def validate_classifier(clf, X_test, y_test):
    print("Valdiating classifier")
    y_pred = clf.predict(X_test)
    labels = sorted(np.unique(y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("Precision Weighted: " + str(precision_score(y_test, y_pred, average='weighted')))
    print("Precision: " + str(precision_score(y_test, y_pred)))
    data.plot_confusion_matrix(cm, classes=labels)


def create_optimized_classifier(X, y, parameters):
    scoring = 'precision'

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    clf = GridSearchCV(xgb.XGBClassifier(), param_grid=parameters, scoring=['accuracy', scoring],
                       cv=cv, verbose=1, refit=scoring, n_jobs=N_JOBS,
                       return_train_score=False, iid=False)
    clf.fit(X, y)
    results = clf.cv_results_
    print("XGB Optimal Model Developed")
    for param, accuracy, score in zip(results['params'], results['mean_test_accuracy'], results['mean_test_' + scoring]):
        print("Accuracy: {:.3f}, {}: {:.3f}, Params: {}".format(accuracy, scoring, score, param))
    print("\nBest {} {}, Params: {}".format(scoring, clf.best_score_, clf.best_params_))
    return clf.best_estimator_


def create_features_pickle(trials, data_loader, clf):
    X, y = data_loader.load(trials)
    clf.fit(X, y)
    features = list(X)
    feature_importances = clf.feature_importances_
    important_feature_idx = np.where(feature_importances)[0]
    features = list(np.array(features)[important_feature_idx])

    pickle_path = CACHE_ROOT + 'features.pickle'

    pickle.dump(from_columns(features), open(pickle_path, "wb"))


if __name__ == '__main__':
    trial_ids = [22, 23, 24, 29, 31, 32, 33, 36, 40, 43]

    dl = data.DataLoader(window_size=100, threshold=1.0, algo_name='maxim')
    clf = xgb.XGBClassifier(n_jobs=N_JOBS)
    create_features_pickle(trial_ids, dl, clf)

    clf = xgb.XGBClassifier(n_jobs=N_JOBS)


