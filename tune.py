import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import data
from data import N_JOBS
import trial_sets


def optimize_classifier(training_ids, validation_ids, data_loader):

    train_iid = True

    X_train, y_train = data_loader.load(training_ids, iid=train_iid)

    X_test, y_test = data_loader.load(validation_ids, iid=False)

    MAX_ESTIMATOR = 1000

    fit_params = {
        "early_stopping_rounds": 50,
        "eval_metric": 'error',
        "eval_set": [[X_test, y_test]],
        "verbose": False
    }

    scoring = 'f1_weighted'

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = xgb.XGBClassifier(n_jobs=N_JOBS, n_estimators=MAX_ESTIMATOR)

    parameters = [
        {'learning_rate': [0.1, 0.01, 0.001, 0.0001]},
        {'max_depth': range(3, 10, 2)},
        {'min_child_weight': range(1, 6, 2)},
        {'gamma': [i / 10.0 for i in range(0, 5)]},
        {'subsample': [i / 10.0 for i in range(6, 10)]},
        {'colsample_bytree': [i / 10.0 for i in range(6, 10)]},
        {'reg_alpha': [1e-6, 1e-7, 1e-8, 1e-5, 1e-2]},
        {'scale_pos_weight': [1, 2, 3]}
    ]

    for parameter in parameters:
        clf = GridSearchCV(model, param_grid=parameter, scoring=scoring,
                           cv=cv, n_jobs=1, iid=train_iid)
        clf.fit(X_train, y_train, **fit_params)

        print("\nParams: {}".format(clf.best_params_))
        print("Num estimators: {}".format(clf.best_estimator_.best_ntree_limit))
        print("{}: {}".format(scoring, clf.best_score_))

        model = clf.best_estimator_
        model.n_estimators = MAX_ESTIMATOR


def tune():
    validation_set_count = 3

    training_ids = trial_sets.top_ids[:-validation_set_count]
    validation_ids = trial_sets.top_ids[-validation_set_count:]

    dl = data.DataLoader(window_size=100, threshold=1.0, algo_name='enhanced', features='comprehensive')
    optimize_classifier(training_ids, validation_ids, dl)


if __name__ == '__main__':
    tune()
