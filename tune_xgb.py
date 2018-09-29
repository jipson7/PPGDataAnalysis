import matplotlib
matplotlib.use('Agg')

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import data

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

CV_FOLDS = 8


def print_model_performance(model, X_train, y_train, X_test, y_test):
    metric = 'map'
    # Fit the algorithm on the data
    model.fit(X_train, y_train, eval_metric=metric)

    # Predict test sets
    test_predictions = model.predict(X_test)
    test_predprob = model.predict_proba(X_test)[:, 1]

    print("Accuracy (Test): {}".format(metrics.accuracy_score(y_test, test_predictions)))
    print("Precision Score (Test): {}".format(metrics.precision_score(y_test, test_predictions, average='weighted')))
    print("AUC Score (Test): {}".format(metrics.roc_auc_score(y_test, test_predprob)))


def tune_nestimators(alg, X_train, y_train,
             early_stopping_rounds=50, metric='map'):
    xgb_param = alg.get_xgb_params()
    xgtrain = xgb.DMatrix(X_train, label=y_train)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=CV_FOLDS,
                      metrics=metric, early_stopping_rounds=early_stopping_rounds, shuffle=True, stratified=True,
                      seed=42)
    alg.set_params(n_estimators=cvresult.shape[0])
    print("Optimal Estimators: {}".format(alg.n_estimators))
    return alg


def tune_params(model, X_train, y_train, param_test):

    gsearch = GridSearchCV(estimator=model, param_grid=param_test,
                            scoring='precision_weighted', iid=False,
                            cv=CV_FOLDS)
    gsearch.fit(X_train, y_train)
    print("Best Params: {}".format(gsearch.best_params_))
    return gsearch.best_estimator_


def tune(X_train, y_train, X_test, y_test):
    xgb1 = XGBClassifier(
        learning_rate=0.015,
        n_estimators=1000,
        max_depth=9,
        min_child_weight=3,
        gamma=0.2,
        subsample=0.8,
        colsample_bytree=0.9,
        objective='binary:logistic',
        nthread=data.N_JOBS,
        scale_pos_weight=3,
        reg_alpha=1e-6,
        seed=27)

    model = tune_nestimators(xgb1, X_train, y_train)

    param_test = {
        'scale_pos_weight': [1, 2, 3, 4]
    }
    model = tune_params(model, X_train, y_train, param_test)

    param_test = {
        'max_depth': range(3, 10, 2),
        'min_child_weight': range(1, 6, 2)
    }
    model = tune_params(model, X_train, y_train, param_test)

    param_test = {
        'gamma':[i/10.0 for i in range(0,5)]
    }
    model = tune_params(model, X_train, y_train, param_test)

    param_test = {
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    model = tune_params(model, X_train, y_train, param_test)

    param_test = {
        'reg_alpha': [1e-6, 1e-7, 1e-8, 1e-5, 1e-2]
    }
    model = tune_params(model, X_train, y_train, param_test)

    print_model_performance(model, X_train, y_train, X_test, y_test)


if __name__ =='__main__':
    dl = data.DataLoader(window_size=100, threshold=1.0, algo_name='maxim', features='comprehensive')
    training_ids = [22, 23, 24, 29, 31, 32, 33]
    test_ids = [36, 40, 43, 22]
    X_train, y_train = dl.load(training_ids)
    X_test, y_test = dl.load(test_ids)
    tune(X_train, y_train, X_test, y_test)
