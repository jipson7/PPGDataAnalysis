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
    trial_id = 16
    devices = data.load_devices(trial_id)

    wrist_device = devices['wrist']

    label_device = devices['reflective']

    print("Aligning timestamps")
    wrist, fingertip = data.normalize_timestamps(wrist_device, label_device)

    print("Extracting Wrist Features")
    fe = data.FeatureExtractor(window_size=100)
    wrist_features = fe.extract_wrist_features(wrist)
    oxygen_labels_pred = fe.extract_label(wrist)

    print("Creating Fingertip labels")
    oxygen_labels_true = fe.extract_label(fingertip)
    assert wrist_features.shape[0] == oxygen_labels_true.shape[0]
    y = fe.create_reliability_label(oxygen_labels_true, oxygen_labels_pred)

    print("Labels created. Label Counts are: ")
    data.print_label_counts(y)

    print("\nRunning Random Forest")
    run_random_forest(wrist_features, y)


if __name__ == '__main__':
    run()

"""
def run_fnn(X_train, y_train, X_test, y_test):
    y = np.concatenate((y_train, y_test))
    encoder = LabelEncoder()
    encoder.fit(y)

    y_train_hot = np_utils.to_categorical(encoder.transform(y_train))

    y_test_hot = np_utils.to_categorical(encoder.transform(y_test))

    EPOCHS = 1
    BATCH_SIZE = 1
    input_dimension = X_train.shape[1]
    num_classes = len(y_train_hot[0])
    # create model
    model = Sequential()
    model.add(Dense(100, input_dim=input_dimension, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Test model
    model.fit(X_train, y_train_hot, epochs=EPOCHS, batch_size=BATCH_SIZE)
    results = model.evaluate(X_test, y_test_hot)
    print(model.metrics_names)
    print(results)
    # print(y_predicted_hot)
    # print("Accuracy: {}".format(accuracy_score(y_test_hot, y_predicted_hot)))
    # print("F1 Score: {}".format(f1_score(y_test_hot, y_predicted_hot, average='weighted')))

"""