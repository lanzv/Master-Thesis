import pandas as pd
import numpy as np
import logging

def load_chants(test_chants_file = "test-chants.csv",
                train_chants_file = "train-chants.csv",
                test_repr_pitch_file = "test-representation-pitch.csv",
                train_repr_pitch_file = "train-representation-pitch.csv"):
    """
    The function will load csv cantus corpus to Panda DataFrames.
    Files going to the function are devided into training and testing parts.
    Both should be filtered by BACOR pipeline.

    Parameters
    ----------
    test_chants_file : str
        path to bacor's test-chants.csv file
    train_chants_file : str
        path to bacor's train-chants.csv file
    test_repr_pitch_file : int
        path to bacor's test-representation-pitch.csv file
    train_repr_pitch_file : int
        path to bacor's train-representation-pitch.csv file
    Returns
    -------
    chants : DataFrame
        dataframe of concatenated training and testing chants datasets
    pitch_representations : DataFrame
        dataframe of concatenated training and testing pitch-representations datasets
    """
    train_chants = pd.read_csv(train_chants_file, index_col='id')
    test_chants = pd.read_csv(test_chants_file, index_col='id')
    chants = pd.concat([train_chants, test_chants])
    pitch_repr_test = pd.read_csv(test_repr_pitch_file, index_col='id')
    pitch_repr_train = pd.read_csv(train_repr_pitch_file, index_col='id')
    pitch_representations = pd.concat([pitch_repr_train, pitch_repr_test])

    return chants, pitch_representations

def prepare_dataset():
    """
    Load {train,test}-chants.csv files and {train,test}-representation-pitch.csv files.
    Return processed chant melodies and its modes.
    The function rely on those csv files to be in working directory.

    Returns
    -------
    X : numpy array
        array of chant melodies
    y : numpy array
        array of chant modes
    """
    chants, pitch_repr = load_chants()
    logging.info("Number of chants: {}".format(len(chants)))
    X, y = [], []
    for segments, mode, id_pitches, id_chant in zip(pitch_repr["1-mer"],
                                                chants['mode'],
                                                pitch_repr.index,
                                                chants.index):
        if not id_pitches == id_chant:
            raise ValueError("IDs of features and modes are not equal!")
        X.append(segments.replace(' ', ''))
        y.append(str(mode))

    return np.array(X), np.array(y)

def load_word_segmentations():
    """
    The function will load word segmentations from cantus corpus using the bacor's prefiltered datasets.
    The function rely on {train,test}-chants.csv files and {train,test}-representation-pitch.csv
    files to be in working directory.

    Returns
    -------
    word_segmentation : list of lists of strings
        word segmentation of each chant, e.g. [["asda", "ddd", "a", "aaa"], ["dddg", "khk"]]
    """
    _, pitch_repr = load_chants()
    word_segmentation = []
    for segments in pitch_repr["words"]:
        word_segmentation.append(segments.split(' '))
    return word_segmentation


def load_syllable_segmentations():
    """
    The function will load syllable segmentations from cantus corpus using the bacor's prefiltered datasets.
    The function rely on {train,test}-chants.csv files and {train,test}-representation-pitch.csv
    files to be in working directory.

    Returns
    -------
    syllable_segmentation : list of lists of strings
        syllable segmentation of each chant, e.g. [["asda", "ddd", "a", "aaa"], ["dddg", "khk"]]
    """
    _, pitch_repr = load_chants()
    syllable_segmentation = []
    for segments in pitch_repr["syllables"]:
        syllable_segmentation.append(segments.split(' '))
    return syllable_segmentation


def load_phrase_segmentations(gregobase_phrases_csv = "./gregobase-chantstrings.csv"):
    """
    The function will load phrase segmentation from preprocessed gregobase dataset.
    Chants have to be converted from gabc format into melody string, where '|' means the end of phrase.

    Parameters
    ----------
    gregobase_phrases_csv : str
        path to preprocessed gregobase chant melodies file with '|' char symbolizes end of phrase
    Returns
    -------
    phrase_segments : list of lists of strings
        phrase segmentation of each chant, e.g. [["asda", "ddd", "a", "aaa"], ["dddg", "khk"]]
    """
    chants = pd.read_csv(gregobase_phrases_csv)
    phrase_segments = []
    for segments in chants["chant_strings"]:
        phrase_segments.append(segments.split('|'))
    return phrase_segments