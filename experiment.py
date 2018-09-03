import data
import pickle
import warnings
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import save_model, load_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score
import xgboost as xgb

# Used to suppress Fscore ill defined
warnings.filterwarnings('ignore')


MODEL_CACHE = './data-cache/models/'


def analyze_classifier(X, y, clf_og, params=None, n_jobs=-1):
    # Note the stratified K fold used by GridSearchCV does NOT shuffle.

    scoring = 'precision_weighted'

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    clf = GridSearchCV(clf_og, param_grid=params, scoring=['accuracy', scoring],
                       cv=cv, verbose=1, refit=scoring, n_jobs=n_jobs,
                       return_train_score=False, iid=False)
    clf.fit(X, y)
    results = clf.cv_results_
    clf_name = type(clf_og).__name__
    print(clf_name + " results:")
    for param, accuracy, score in zip(results['params'], results['mean_test_accuracy'], results['mean_test_' + scoring]):
        print("Accuracy: {:.3f}, {}: {:.3f}, Params: {}".format(accuracy, scoring, score, param))
    print("\nBest {} {}, Params: {}".format(scoring, clf.best_score_, clf.best_params_))
    estimator = clf.best_estimator_
    pickle_path = MODEL_CACHE + clf_name
    if clf_name == "KerasClassifier":
        save_model(estimator.model, pickle_path)
    else:
        pickle.dump(estimator, open(pickle_path, "wb"))


def run_random_forest(X, y):
    print("\nRunning Random Forest")
    # parameters = {
    #     'n_estimators': [x for x in range(20, 100, 10)],
    #     'criterion': ['gini']
    # }
    parameters = {
        'n_estimators': [90],
        'criterion': ['gini']
    }
    analyze_classifier(X, y, RandomForestClassifier(), params=parameters)


def run_logistic_regression(X, y):
    print("\nRunning Logistic Regression")
    # parameters = {
    #     'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
    #     'penalty': ['l2']
    # }
    parameters = {
        'C': [0.00001],
        'penalty': ['l2']
    }
    analyze_classifier(X, y, LogisticRegression(), params=parameters)


def run_svc(X, y):
    print("\nRunning Support Vector Classification")
    # parameters = {
    #     'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    #     'kernel': ['rbf', 'sigmoid']
    # }
    parameters = {
        'C': [10],
        'kernel': ['rbf']
    }
    analyze_classifier(X, y, SVC(), params=parameters)


def run_gradient_boost(X, y):
    print("\nRunning Gradient Boost Classifier")
    parameters = {
        'loss': ['deviance', 'exponential'],
        'criterion': ['friedman_mse', 'mse']
    }
    analyze_classifier(X, y, GradientBoostingClassifier(), params=parameters)


def run_nn(X, y):
    print("\nRunning NN")
    from neural_network import get_model_generator
    # parameters = {
    #     'hidden_layers': [1, 2, 3, 4],
    #     'hidden_layer_size': [4, 8, 12, 16, 20],
    #     'optimizer': ['rmsprop', 'adam'],
    #     'init': ['glorot_uniform', 'normal', 'uniform'],
    #     'epochs': [1],
    #     'batch_size': [32]
    # }
    parameters = {
        'hidden_layers': [1],
        'hidden_layer_size': [8],
        'optimizer': ['rmsprop'],
        'init': ['normal'],
        'epochs': [1],
        'batch_size': [32]
    }
    model_gen = get_model_generator(X.shape[1])
    clf = KerasClassifier(build_fn=model_gen, verbose=0)
    analyze_classifier(X, y, clf, params=parameters, n_jobs=1)


def run_xg_boost(X, y):
    print("\nRunning Gradient Boost Classifier")
    parameters = {
        'booster': ['gbtree', 'dart', 'gblinear'],
        'learning_rate': [0, 0.25, 0.5, 0.75, 1],
        'n_estimators': [x for x in range(10, 100, 10)],
        'objective': ['binary:logistic', 'binary:logitraw', 'binary:hinge'],
        'n_jobs': [4]
    }
    parameters={}
    analyze_classifier(X, y, xgb.XGBClassifier(), params=parameters, n_jobs=1)


def create_optimized_models(trial_ids):
    X, y = load_data(trial_ids)
    run_xg_boost(X, y)
    # run_gradient_boost(X, y)
    # run_random_forest(X, y)
    # run_logistic_regression(X, y)
    # run_nn(X, y)
    # run_svc(X, y)


def apply_model(model_name, trial_ids):
    model_path = MODEL_CACHE + model_name
    if model_name == "KerasClassifier":
        clf = load_model(model_path)
    else:
        clf = pickle.load(open(model_path, "rb"))

    if hasattr(clf, "feature_importances_"):
        print("Feature importances: ")
        print(clf.feature_importances_)
    for trial_id in trial_ids:
        X, y_true = load_data([trial_id])
        if model_name == "KerasClassifier":
            y_pred = clf.predict_classes(X).tolist()
        else:
            y_pred = clf.predict(X)
        labels = sorted(np.unique(y_pred))
        print("\nReport for {} run against trial {} :".format(model_name, trial_id))
        print("Label set: " + str(labels))
        print("Precision: " + str(precision_score(y_true, y_pred, average='weighted')))
        cm = confusion_matrix(y_true, y_pred)
        data.plot_confusion_matrix(cm, classes=labels, title=model_name + " Confusion Matrix")
        plt.show()


def load_data(trial_ids):
    X_s = []
    y_s = []
    for trial_id in trial_ids:
        devices = data.load_devices(trial_id, algo_name="enhanced")

        print("Extracting Wrist Features")
        fe = data.FeatureExtractor(window_size=100)
        X = fe.extract_wrist_features(devices[0])

        print("Creating Reliability labels")
        y = fe.create_reliability_label(devices, threshold=2.0)
        data.print_label_counts(y)

        X_s.append(X)
        y_s.append(y)
    X = np.vstack(X_s)
    y = np.concatenate(y_s)
    return X, y


if __name__ == '__main__':
    trial_ids = data.list_trials()
    training_trials = [13, 18, 20]
    create_optimized_models(training_trials)

    testing_trials = [22]
    apply_model("XGBClassifier", testing_trials)
