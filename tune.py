import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import data
from data import N_JOBS


def optimize_classifier(training_ids, validation_ids, data_loader):
    X_train, y_train = data_loader.load(training_ids)

    X_test, y_test = data_loader.load(validation_ids)

    parameters = {
        'learning_rate': [0.05, 0.1, 0.3],
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9]
    }

    fit_params = {
        "early_stopping_rounds": 30,
        "eval_metric": 'map',
        "eval_set": [[X_test, y_test]]
    }

    scoring = 'precision_weighted'

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # TODO check if I need IID = False?
    clf = GridSearchCV(xgb.XGBClassifier(), param_grid=parameters, scoring=scoring,
                       cv=cv, verbose=1, refit=False, n_jobs=N_JOBS, iid=False)

    clf.fit(X_train, y_train, **fit_params)

    print("\nBest {} {}, Params: {}".format(scoring, clf.best_score_, clf.best_params_))


def tune():
    # training_ids = [22, 23, 24, 29, 31, 32, 33, 36, 40, 43]
    training_ids = [24, 29, 31, 32, 33, 36, 40, 43]
    validation_ids = [22, 23]
    dl = data.DataLoader(window_size=100, threshold=1.0, algo_name='maxim', features='comprehensive')
    optimize_classifier(training_ids, validation_ids, dl)


if __name__ == '__main__':
    tune()
