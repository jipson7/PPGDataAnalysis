from data import N_JOBS


xgboost = {
    'n_jobs': N_JOBS,
    'learning_rate': 0.1,
    'max_depth': 3,
    'n_estimators': 200
}

gbc = {
    'learning_rate': 0.1,
    'n_estimators': 1200,
    'max_depth': 3,
    'subsample': 0.5,
    'min_samples_leaf': 1
}