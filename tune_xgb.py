import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

# TODO Extract to models version
from sklearn import cross_validation, metrics
from sklearn.model_selection import GridSearchCV
import data

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


def modelfit(alg, X_train, y_train, X_test, y_test,
             useTrainCV=True, cv_folds=5,
             early_stopping_rounds=50, metric='map'):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics=metric, early_stopping_rounds=early_stopping_rounds, show_stdv=True)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(X_train, y_train, eval_metric=metric)

    # Predict training set:
    train_predictions = alg.predict(X_train)
    train_predprob = alg.predict_proba(X_train)[:, 1]

    # Predict test sets
    test_predictions = alg.predict(X_test)
    test_predprob = alg.predict_proba(X_test)[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy (Train): {}".format(metrics.accuracy_score(y_train, train_predictions)))
    print("{} Score (Train): {}".format(metric, metrics.roc_auc_score(y_train, train_predprob)))

    print("Accuracy (Test): {}".format(metrics.accuracy_score(y_test, test_predictions)))
    print("{} Score (Test): {}".format(metric, metrics.roc_auc_score(y_test, test_predprob)))

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)

    # get top 100 features
    feat_top = feat_imp[:25]

    feat_top.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.savefig("./local-cache/experiments/tune_xgb.png")


def tune_1(X_train, y_train, X_test, y_test):
    xgb1 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
    modelfit(xgb1, X_train, y_train, X_test, y_test)


if __name__ =='__main__':
    dl = data.DataLoader(window_size=100, threshold=1.0, algo_name='maxim', features='comprehensive')
    training_ids = [24, 29, 31, 32, 33, 36, 40, 43]
    test_ids = [22, 23]
    X_train, y_train = dl.load(training_ids)
    X_test, y_test = dl.load(test_ids)
    tune_1(X_train, y_train, X_test, y_test)
