from Levenshtein import distance
import numpy as np
import sys

def vocab_levenshtein_score(segmented_chants):
    """
    First, get the vocabulary from segmented chants.
    Second, compute levenhstein distances of each pair of vocabulary segments.
    The score compute as 1/N sum_{vocab_segment}(min(levenshtein distances with between vocab segment and all rest vocab segments))

    Parameters
    ----------
    segmented_chants : list of list of strings
        list of chants represented as list of string segments
    Returns
    -------
    levenshtein_score : float
        1/N sum_{vocab_segment}(min(levenshtein distances with between vocab segment and all rest vocab segments))
    """
    vocab = set()
    for chant in segmented_chants:
        for segment in chant:
            vocab.add(segment)
    vocab = list(vocab)
    distance_matrix = np.zeros((len(vocab), len(vocab)))
    for i in range(len(vocab)):
        distance_matrix[i, i] = sys.maxsize
        for j in range(i+1, len(vocab)):
            i_j_distance = distance(vocab[i], vocab[j])
            distance_matrix[i, j] = i_j_distance
            distance_matrix[j ,i] = i_j_distance
    levenshtein_score = np.average(np.amin(distance_matrix, axis=0))
    return levenshtein_score