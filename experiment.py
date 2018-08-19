import data
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def run_random_forest(X, y):
    parameters = {
        'n_estimators': [x for x in range(20, 50)],
        'criterion': ['gini']
    }
    scoring = ['accuracy', 'f1']
    clf = GridSearchCV(RandomForestClassifier(),
                       parameters, scoring=scoring,
                       cv=5, verbose=1, refit='accuracy')
    clf.fit(X, y)
    print("\nAccuracy {}, Params: {}".format(clf.best_score_, clf.best_params_))


def run():
    data.list_trials()
    trial_id = 15
    devices = data.load_devices(trial_id)

    print("Aligning timestamps")
    wrist, fingertip = data.normalize_timestamps(devices['wrist'], devices['reflective'])

    print("Extracting Wrist Features")
    fe = data.FeatureExtractor(window_size=100)
    wrist_features = fe.extract_wrist_features(wrist)

    print("Creating labels")
    oxygen_labels_true = fe.extract_label(fingertip)
    oxygen_labels_pred = fe.extract_label(wrist)
    assert wrist_features.shape[0] == oxygen_labels_true.shape[0]
    y = fe.create_reliability_label(oxygen_labels_true, oxygen_labels_pred)

    print("Labels created. Label Counts are: ")
    data.print_label_counts(y)

    print("\nRunning Random Forest GridSearch")
    run_random_forest(wrist_features, y)

if __name__ == '__main__':
    run()