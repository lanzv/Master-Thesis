from Levenshtein import distance
import numpy as np
import logging

def vocab_levenshtein_score(segmented_chants):
    """
    First, get the vocabulary from segmented chants.
    Second, compute levenhstein distances of each pair of vocabulary segments.
    The score compute as 1/N sum_{vocab_segment}(median(levenshtein distances between vocab 
                                    segment and all other vocab segments with same size))

    Parameters
    ----------
    segmented_chants : list of list of strings
        list of chants represented as list of string segments
    Returns
    -------
    levenshtein_score : float
        1/N sum_{vocab_segment}(median(levenshtein distances between vocab
                                    segment and all other vocab segments with same size))
    """
    median_sum = 0
    segment_count = 0
    vocab_sizes = {}
    for chant in segmented_chants:
        for segment in chant:
            if len(segment) in vocab_sizes:
                vocab_sizes[len(segment)].add(segment)
            else:
                vocab_sizes[len(segment)] = {segment}
    for size in vocab_sizes:
        vocab = list(vocab_sizes[size])
        if len(vocab) <= 1:
            logging.warn("There is a ignored {} segments with size {} when computing levenhstein score.".format(len(vocab), size))
            continue
        segment_count += len(vocab)
        distance_matrix = np.zeros((len(vocab), len(vocab)))
        for i in range(len(vocab)):
            for j in range(i+1, len(vocab)):
                i_j_distance = distance(vocab[i], vocab[j])
                distance_matrix[i, j] = i_j_distance
                distance_matrix[j ,i] = i_j_distance
        median_sum += np.sum(
            np.nanmedian(
                np.where(distance_matrix == 0.0, np.nan, distance_matrix)
                , axis=0
                )
            )
    levenshtein_score = median_sum/segment_count
    return levenshtein_score