import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import random

def list2string(segmented_chants):
    """
    Convert chant segments in list to chant segments in string for all chants.
    New chant representation is a string of segments separated by spaces.
    E.g. [["aaa", "bbb", "ccc"]] -> ["aaa bbb ccc"]

    Parameters
    ----------
    segmented_chants : list of lists of strings
        list of chants, each chant is represented as list of segments
    Returns
    -------
    string_segmentations : list of strings
        list of chants, each chant is represented as string, where segments are separated by spaces
    """
    string_segmentations = []
    for chant_segments in segmented_chants:
        string_segmentations.append(' '.join(chant_segments))
    return string_segmentations




def get_bacor_model(all_features_vectorizer=False):
    """
    Get the BACOR Linear SVC model with their hyperparameters and TfidfVectorizer.
    Hyperparameters are not tuned. We use this model only for fast iteraion evaluation.
    For the final evlauation score is used the BACOR model version which is tuned.
    
    Parameters
    ----------
    all_features_vectorizer : boolean
        true to not limit number of features considered by vectorizer
        false when using 5000 as max features
    Returns
    -------
    pipeline : Pipeline
        sklearn pipeline of TfidfVectorizer and LinearSVC, basically pipeline of BACOR model
    """
    if all_features_vectorizer:
        tfidf_params = dict(
            # Defaults
            strip_accents=None,
            stop_words=None,
            ngram_range=(1,1),
            max_df=1.0,
            min_df=1,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
            lowercase=False,
            analyzer='word',
            token_pattern=r'[^ ]+')
    else:
        tfidf_params = dict(
            # Defaults
            strip_accents=None,
            stop_words=None,
            ngram_range=(1,1),
            max_df=1.0,
            min_df=1,
            max_features=5000,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
            lowercase=False,
            analyzer='word',
            token_pattern=r'[^ ]+')
    svc_params = {
        'penalty': 'l2',
        'loss': 'squared_hinge',
        'multi_class': 'ovr',
        'random_state': np.random.randint(100)
    }
    return Pipeline([
        ('vect', TfidfVectorizer(**tfidf_params)),
        ('clf', LinearSVC(**svc_params)),
    ])

def get_nb_model(all_features_vectorizer = False):
    """
    Get Multinomial Naive Bayes model with the Tfidf Vectorizer as part of the pipeline.

    Parameters
    ----------
    all_features_vectorizer : boolean
        true to not limit number of features considered by vectorizer
        false when using 5000 as max features
    Returns
    -------
    pipeline : Pipeline
        sklearn pipeline of TfidfVectorizer and Multinomial Naive Bayes
    """
    if all_features_vectorizer:
        pipeline = Pipeline([('tfidf', TfidfVectorizer(strip_accents=None,
                    stop_words=None,
                    ngram_range=(1,1),
                    max_df=1.0,
                    min_df=1,
                    use_idf=True,
                    smooth_idf=True,
                    sublinear_tf=False,
                    lowercase=False,
                    analyzer='word',
                    token_pattern=r'[^ ]+')),
                    ('clf', MultinomialNB(alpha=0))])
    else:
        pipeline = Pipeline([('tfidf', TfidfVectorizer(strip_accents=None,
            stop_words=None,
            ngram_range=(1,1),
            max_df=1.0,
            min_df=1,
            max_features=5000,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
            lowercase=False,
            analyzer='word',
            token_pattern=r'[^ ]+')),
            ('clf', MultinomialNB(alpha=0))])
    return pipeline

def get_bacor_nottuned_scores(train_chants, train_modes, test_chants, test_modes):
    """
    Convert chant segments to string representation and train the bacor model without tuning.
    Compute accuracy and f1 of test dataset.

    Parameters
    ---------
    train_chants : list of list of strings
        train chants represented as list of string segments
    train_modes : list of strings
        train modes
    test_chants : list of list of strings
        test chants represented as list of string segments
    test_modes : list of strings
        test modes
    Returns
    -------
    test_accuracy : float
        accuracy score of testing dataset of BACOR SVC model
    test_f1 : float
        f1 score of testing dataset of BACOR SVC model
    """
    train_data, test_data = list2string(train_chants), list2string(test_chants)
    bacor_model = get_bacor_model()
    bacor_model.fit(train_data, train_modes)
    test_predictions = bacor_model.predict(test_data)
    test_accuracy = accuracy_score(test_modes, test_predictions)
    test_f1 = f1_score(test_modes, test_predictions, average='weighted')
    return test_accuracy, test_f1



def get_topsegments_frequency(segmented_chants, modes,
                            top_segments: list, ignore_segments: bool = False):
    """
    Get frequency dataframe of top segments (that could be get from feature extraction methods)
    over all training and testing segmented chants. The freuqency tells the dominancy of some
    modes considering top segments (top SVC features).

    Parameters
    ----------
    segmented_chants : list of lists of strings
        list of segmented chants represented as list of segments
    modes : list of lists of chars
        list of chant modes
    top_segments : list of strings
        list of segments that are chosen from feature selection pipeline
    ignore_segments : bool
        in case of True, check segments frequency based on segmentation
        in case of False, check segments frequency based on the occurencies in the original melody
            (where the knowledge of the segmentation is not used)
    Returns
    -------
    df : DataFrame
        dataframe of frequency_matrix of segments and their frequencies in modes that should sum up to 1
        considering one segment and all modes, index is a mode list ["1", .., "8"], columns is a list of
        segments with the information of the overall occurence number over all modes
    """
    top_segment_set = set(top_segments)
    melody_frequencies = {}
    for segment in top_segment_set:
        melody_frequencies[segment] = {}
    # collect data
    for segments, mode in zip(segmented_chants, modes):
        if ignore_segments:
            chant = ''.join(segments)
            for melody in top_segment_set:
                if melody in chant:
                    if not mode in melody_frequencies[melody]:
                        melody_frequencies[melody][mode] = 0
                    melody_frequencies[melody][mode] += 1
        else:
            for segment in segments:
                if segment in top_segment_set:
                    if not mode in melody_frequencies[segment]:
                        melody_frequencies[segment][mode] = 0
                    melody_frequencies[segment][mode] += 1



    # Create DataFrame
    index = ["1", "2", "3", "4", "5", "6", "7", "8"]
    columns = []
    frequency_matrix = np.zeros((len(index), len(top_segments)))
    for i, melody in enumerate(top_segments):
        for j, mode in enumerate(index):
            if mode in melody_frequencies[melody]:
                frequency_matrix[j, i] = melody_frequencies[melody][mode]
        # store counts
        columns.append(melody + "(" + str(int(np.sum(frequency_matrix[:, i]))) + ")")
        # normalize to sum up over modes to 1
        frequency_matrix[:, i] = frequency_matrix[:, i]/np.sum(frequency_matrix[:, i])


    df = DataFrame(frequency_matrix, index=index, columns=columns)

    return df


def get_random_reduced_list(array: list, k: int):
    """
    Create a copy of the comming list and remove from it k random elements.

    Parameters
    ----------
    array : list
        the original list we want the reduced version of
    k : int
        number of elements to randomly ignore
    Returns
    -------
    final_array : list
        reduced array by k random elements
    """
    final_array = array.copy()
    for _ in range(k):
        rand_ind =  random.randint(0, len(final_array)-1)
        final_array = [ele for ele_ind, ele in enumerate(final_array) if ele_ind != rand_ind]
    return final_array