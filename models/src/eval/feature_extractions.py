
from sklearn.feature_selection import SelectFromModel
import logging
from sklearn.feature_selection import SequentialFeatureSelector
from src.utils.eval_helpers import list2string, get_bacor_model
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd

# BACOR feature selection functions
# model is a pipeline of TFIDF vectorizer and SVC linear classifier)

def features_from_model(X_train, y_train, X_test, y_test, max_features = None):
    bacor_model = get_bacor_model()

    selector = SelectFromModel(estimator=bacor_model.steps[1][1], max_features = max_features)
    X_vectorized = bacor_model.steps[0][1].fit_transform(X_train)
    X_df = pd.DataFrame(X_vectorized.toarray(), columns = bacor_model.steps[0][1].get_feature_names_out())
    selector.fit(X_df, y_train)

    # Extract top features
    selected_features = set(selector.get_feature_names_out())
    logging.info("From model approach - Selected features: {} (only 10% of them will be chosen)".format(len(selected_features)))
    melodies = {}
    for chant in X_train:
        for segment in chant.split():
            if segment in selected_features:
                if segment in melodies:
                    melodies[segment] += 1
                else:
                    melodies[segment] = 1
    top_melodies = sorted(melodies, key=melodies.get, reverse=True)
    logging.info("From model approach Train data - First feature occurences: {} , Last feature occurences: {}".format(melodies[top_melodies[0]], melodies[top_melodies[-1]]))


    # Prepare reduced datasets
    X_train_new, X_test_new = [], []
    for chant in X_train:
        new_chant = []
        for segment in chant.split():
            if segment in selected_features:
                new_chant.append(segment)
        X_train_new.append(new_chant)
    for chant in X_test:
        new_chant = []
        for segment in chant.split():
            if segment in selected_features:
                new_chant.append(segment)
        X_test_new.append(new_chant)

    # Evaluate bacor_model on reduced features
    train_data, test_data = list2string(X_train_new), list2string(X_test_new)
    bacor_model = get_bacor_model()
    bacor_model.fit(train_data, y_train)
    predictions = bacor_model.predict(test_data)
    logging.info("From model approach - reduced accuracy: {:.2f}%, reduced f1: {:.2f}%".format(
        accuracy_score(y_test, predictions)*100, f1_score(y_test, predictions, average='weighted')*100
    ))
    return top_melodies


def features_by_additativ_approach(X_train, y_train, X_test, y_test, max_features = "auto"):
    # Define bacor model without tuning
    bacor_model = get_bacor_model()

    # Init and fit feature selector
    X_vectorized = bacor_model.steps[0][1].fit_transform(X_train)
    X_df = pd.DataFrame(X_vectorized.toarray(), columns = bacor_model.steps[0][1].get_feature_names_out())
    selector = SequentialFeatureSelector(estimator = bacor_model.steps[1][1],
                                    n_features_to_select=max_features,
                                    direction='forward')
    selector.fit(X_df, y_train)

    # Extract top features
    selected_features = set(selector.get_feature_names_out())
    logging.info("Additativ approach - Selected features: {}".format(len(selected_features)))
    melodies = {}
    for chant in X_train:
        for segment in chant.split():
            if segment in selected_features:
                if segment in melodies:
                    melodies[segment] += 1
                else:
                    melodies[segment] = 1
    top_melodies = sorted(melodies, key=melodies.get, reverse=True)
    logging.info("Additativ approach Train data - First feature occurences: {} , Last feature occurences: {}".format(melodies[top_melodies[0]], melodies[top_melodies[-1]]))

    # Prepare reduced datasets
    X_train_new, X_test_new = [], []
    for chant in X_train:
        new_chant = []
        for segment in chant.split():
            if segment in selected_features:
                new_chant.append(segment)
        X_train_new.append(new_chant)
    for chant in X_test:
        new_chant = []
        for segment in chant.split():
            if segment in selected_features:
                new_chant.append(segment)
        X_test_new.append(new_chant)

    # Evaluate bacor_model on reduced features
    train_data, test_data = list2string(X_train_new), list2string(X_test_new)
    bacor_model = get_bacor_model()
    bacor_model.fit(train_data, y_train)
    predictions = bacor_model.predict(test_data)
    logging.info("Additative approach - reduced accuracy: {:.2f}%, reduced f1: {:.2f}%".format(
        accuracy_score(y_test, predictions)*100, f1_score(y_test, predictions, average='weighted')*100
    ))


    return top_melodies