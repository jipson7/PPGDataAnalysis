import data
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score
import xgboost as xgb
import warnings
import matplotlib.pyplot as plt
from data import N_JOBS, CM_CACHE

warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)


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


def create_average_cm(clf, trial_ids, data_loader):
    cms = []
    for trial_id in trial_ids:
        training_ids = trial_ids.copy()
        training_ids.remove(trial_id)
        cm = create_cm(clf, training_ids, trial_id, data_loader)
        cms.append(cm)
    avg_cm = np.average(cms, axis=0)
    avg_cm = np.array(avg_cm).astype(int)
    data.plot_confusion_matrix(avg_cm)
    plt.savefig(CM_CACHE + 'cm-' + str(data_loader) + '.png')


def create_cm(clf, training_ids, test_id, data_loader):
    X_train, y_train = data_loader.load(training_ids)
    clf.fit(X_train, y_train)
    X_test, y_test = data_loader.load([test_id])
    y_pred = clf.predict(X_test)
    print("Trial {} results: ".format(test_id))
    print("Precision Weighted: " + str(precision_score(y_test, y_pred, average='weighted')))
    print("Precision: " + str(precision_score(y_test, y_pred)))
    return confusion_matrix(y_test, y_pred)


if __name__ == '__main__':
    trial_ids = [22, 23, 24, 29, 31, 32, 33, 36, 40, 43]

    clf = xgb.XGBClassifier(n_jobs=N_JOBS)
    dl = data.DataLoader(window_size=100, threshold=1.0, algo_name='maxim')

    create_average_cm(clf, trial_ids, dl)


