import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import data
from data import N_JOBS


def optimize_classifier(trial_ids, data_loader):
    X, y = data_loader.load(trial_ids)
    parameters = {
        'learning_rate': [0.1, 0.3, 0.5],
        'n_estimators': [100, 1000],
        'max_depth': [3, 6, 9]
    }
    scoring = 'precision_weighted'

    clf = GridSearchCV(xgb.XGBClassifier(), param_grid=parameters, scoring=scoring,
                       cv=2, verbose=3, refit=False, n_jobs=N_JOBS,
                       return_train_score=False, iid=False)
    clf.fit(X, y)

    print("\nBest {} {}, Params: {}".format(scoring, clf.best_score_, clf.best_params_))


def tune():
    trial_ids = [22, 23]
    dl = data.DataLoader(window_size=100, threshold=1.0, algo_name='maxim', features='comprehensive')
    optimize_classifier(trial_ids, dl)


if __name__ == '__main__':
    tune()