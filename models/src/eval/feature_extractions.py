
from sklearn.feature_selection import SelectFromModel
import logging
from sklearn.feature_selection import SequentialFeatureSelector
from src.utils.eval_helpers import list2string, get_bacor_model, get_nb_model
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import numpy as np
from src.utils.plotters import plot_topsegments_densities
from decimal import Decimal, ROUND_HALF_UP

# BACOR feature selection functions
# model is a pipeline of TFIDF vectorizer and SVC linear classifier)

def features_from_model(X_train, y_train, X_test, y_test, max_features = None, occurence_coef=10, all_features_vectorizer = False):
    """
    Fit the SVC model, with no tuning, find the top max_features*occurence_coef features with highest coeficients.
    Take the top max_features of them with the higher occurence number in trainin dataset.
    Also try to fit the model again with reduced features by these new top_melodies and log the result.

    Parameters
    ----------
    X_train : list of strings
        list of train chants represented as a string with segments separated by spaces
    y_train :  list of strings
        list of training modes
    X_test : list of strings
        list of train chants represented as a string with segments separated by spaces
    y_test : list of strings
        list of testing modes
    max_features : int
        maximum number of features to get from feature extraction
    occurence_coef :  int
        find the fe_occurence_coef times more best features from model then max_features_from_model says, 
        after that pick only max_features_from_model features with the most occurences
    all_features_vectorizer : boolean
        true to not limit number of features considered by vectorizer
        false when using 5000 as max features
    Returns
    -------
    top_melodies : list of strings
        list of top max_features or len(vocabulary) melodies selected by this method
    """
    bacor_model = get_bacor_model(all_features_vectorizer)

    if max_features != None:
        max_features *= occurence_coef

    selector = SelectFromModel(estimator=bacor_model.steps[1][1], max_features = max_features)
    X_vectorized = bacor_model.steps[0][1].fit_transform(X_train)
    X_df = pd.DataFrame(X_vectorized.toarray(), columns = bacor_model.steps[0][1].get_feature_names_out())
    selector.fit(X_df, y_train)

    # Extract top features
    selected_features = set(selector.get_feature_names_out())
    if max_features != None:
        logging.info("From model approach - Selected features: {} (only {} of them will be chosen)".format(len(selected_features), int(max_features/occurence_coef)))
    melodies = {}
    for chant in X_train:
        for segment in chant.split():
            if segment in selected_features:
                if segment in melodies:
                    melodies[segment] += 1
                else:
                    melodies[segment] = 1
    if max_features == None:
        top_melodies = sorted(melodies, key=melodies.get, reverse=True)
    else:
        top_melodies = sorted(melodies, key=melodies.get, reverse=True)[:int(max_features/occurence_coef)]
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
    bacor_model = get_bacor_model(all_features_vectorizer)
    bacor_model.fit(train_data, y_train)
    bacor_predictions = bacor_model.predict(test_data)
    nb_model = get_nb_model(all_features_vectorizer)
    nb_model.fit(train_data, y_train)
    nb_predictions = nb_model.predict(test_data)
    logging.info("From model approach - reduced bacor accuracy: {:.2f}%, reduced bacor f1: {:.2f}% - reduced NB accuracy: {:.2f}%, reduced NB f1: {:.2f}%".format(
        accuracy_score(y_test, bacor_predictions)*100, f1_score(y_test, bacor_predictions, average='weighted')*100,
        accuracy_score(y_test, nb_predictions)*100, f1_score(y_test, nb_predictions, average='weighted')*100
    ))
    return top_melodies


def features_by_additativ_approach(X_train, y_train, X_test, y_test, max_features = "auto"):
    """
    Fit the SVC model, with no tuning, find the top max_features using additative feature extraction.
    The model will try all posibilities and pick the one with the highest final score. Then, the model
    tries to find the next feature ect...
    Also try to fit the model again with reduced features by these new top_melodies and log the result.

    Parameters
    ----------
    X_train : list of strings
        list of train chants represented as a string with segments separated by spaces
    y_train :  list of strings
        list of training modes
    X_test : list of strings
        list of train chants represented as a string with segments separated by spaces
    y_test : list of strings
        list of testing modes
    max_features : int
        maximum number of features to get from feature extraction
    Returns
    -------
    top_melodies : list of strings
        list of top max_features or len(vocabulary) melodies selected by this method
    """
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




def show_topsegments_densities(segmented_chants: list, modes: list, topsegments: set, mode_list = ["1", "2", "3", "4", "5", "6", "7", "8"]):
    """
    Compute percentage of top important segments in the chant position over all modes.
    Plot charts visualizaing positions of important segments.

    Parameters
    ----------
    segmented_chants: list of list of strings
        list of segmented chants represented as a list of segments
        example: [["asda", "asdasd", "as", "ds"]]
    modes : list of strings
        list of modes
    topsegments : set of strings
        top segments we want density charts of
    mode_list : list of strings
        list of modes we are considering
    """
    # Get Density Data
    # Prepare Density
    densities = {}
    num_chants = {}
    densities_scale = 400
    for mode in mode_list:
        num_chants[mode] = 0
        densities[mode] = np.zeros((densities_scale)) # 100% in 400 cells -> 4 cells ~ 1%
    # Get Percentage distribution of unique segments
    for chant, mode in zip(segmented_chants, modes):
        chant_len = len(''.join(chant))
        actual_position = 0
        tone_size = float(densities_scale)/float(chant_len)
        segment_pointer = 0
        num_chants[mode] += 1
        for i in range(1, densities_scale+1):
            while i > Decimal((actual_position + len(chant[segment_pointer]))*tone_size).quantize(0, ROUND_HALF_UP):
                actual_position += len(chant[segment_pointer])
                segment_pointer += 1
            if chant[segment_pointer] in topsegments:
                densities[mode][i-1] += 1
    for mode in mode_list:
        densities[mode] /= num_chants[mode]
        densities[mode] *= 100

    plot_topsegments_densities(densities, mode_list)