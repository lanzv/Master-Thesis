import pandas as pd
import numpy as np
import logging

def load_chants(test_chants_file = "test-chants.csv",
                train_chants_file = "train-chants.csv",
                test_repr_pitch_file = "test-representation-pitch.csv",
                train_repr_pitch_file = "train-representation-pitch.csv"):
    train_chants = pd.read_csv(train_chants_file, index_col='id')
    test_chants = pd.read_csv(test_chants_file, index_col='id')
    logging.info("Number of train chants: {}".format(len(train_chants)))
    logging.info("Number of test chants: {}".format(len(test_chants)))
    chants = pd.concat([train_chants, test_chants])
    pitch_repr_test = pd.read_csv(test_repr_pitch_file, index_col='id')
    pitch_repr_train = pd.read_csv(train_repr_pitch_file, index_col='id')
    pitch_representations = pd.concat([pitch_repr_train, pitch_repr_test])

    return chants, pitch_representations

def prepare_dataset():
    chants, pitch_repr = load_chants()
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
    _, pitch_repr = load_chants()
    word_segmentation = []
    for segments in pitch_repr["words"]:
        word_segmentation.append(segments.split(' '))
    return word_segmentation