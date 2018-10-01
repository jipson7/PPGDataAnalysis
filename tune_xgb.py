import matplotlib
matplotlib.use('Agg')

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import data

CV_FOLDS = 8


def print_model_performance(model, X_train, y_train, X_test, y_test):
    metric = 'map'
    # Fit the algorithm on the data
    model.fit(X_train, y_train, eval_metric=metric)

    train_predictions = model.predict(X_train)

    # Predict test sets
    test_predictions = model.predict(X_test)

    print("Accuracy (Train): {}".format(metrics.accuracy_score(y_train, train_predictions)))
    print("Precision Score (Train): {}".format(metrics.precision_score(y_train, train_predictions, average='weighted')))

    print("Accuracy (Test): {}".format(metrics.accuracy_score(y_test, test_predictions)))
    print("Precision Score (Test): {}".format(metrics.precision_score(y_test, test_predictions, average='weighted')))



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
        learning_rate=0.1,
        n_estimators=1000,
        nthread=data.N_JOBS,
        seed=42)

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
    train_loader = data.DataLoader(threshold=2.0, algo_name='enhanced')
    test_loader = data.DataLoader(threshold=2.0, algo_name='maxim')
    enhanced_ids = [13, 20, 22, 24]
    test_ids = [23, 32]
    X_train, y_train = train_loader.load(enhanced_ids, iid=True)
    X_test, y_test = test_loader.load(test_ids, iid=False)
    tune(X_train, y_train, X_test, y_test)
