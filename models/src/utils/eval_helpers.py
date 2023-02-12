import numpy as np
from pandas import DataFrame

def list2string(segmented_chants):
    """
    [["aaa", "bbb", "ccc"]] -> ["aaa bbb ccc"]
    """
    string_segmentations = []
    for chant_segments in segmented_chants:
        string_segmentations.append(' '.join(chant_segments))
    return string_segmentations


def get_topmelodies_frequency(train_segmented_chants, train_modes,
                            test_segmented_chants, test_modes,
                            top_melodies: list, ignore_segments: bool = False):
    top_melodies_set = set(top_melodies)
    melody_frequencies = {}
    # collect training data
    for segments, mode in zip(train_segmented_chants, train_modes):
        if ignore_segments:
            chant = ''.join(segments)
            for melody in top_melodies_set:
                if melody in chant:
                    if not melody in melody_frequencies:
                        melody_frequencies[melody] = {}
                    if not mode in melody_frequencies[melody]:
                        melody_frequencies[melody][mode] = 0
                    melody_frequencies[melody][mode] += 1
        else:
            for segment in segments:
                if segment in top_melodies_set:
                    if not segment in melody_frequencies:
                        melody_frequencies[segment] = {}
                    if not mode in melody_frequencies[segment]:
                        melody_frequencies[segment][mode] = 0
                    melody_frequencies[segment][mode] += 1

    # collect test data
    for segments, mode in zip(test_segmented_chants, test_modes):
        if ignore_segments:
            chant = ''.join(segments)
            for melody in top_melodies_set:
                if melody in chant:
                    if not melody in melody_frequencies:
                        melody_frequencies[melody] = {}
                    if not mode in melody_frequencies[melody]:
                        melody_frequencies[melody][mode] = 0
                    melody_frequencies[melody][mode] += 1
        else:
            for segment in segments:
                if segment in top_melodies_set:
                    if not segment in melody_frequencies:
                        melody_frequencies[segment] = {}
                    if not mode in melody_frequencies[segment]:
                        melody_frequencies[segment][mode] = 0
                    melody_frequencies[segment][mode] += 1



    # Create DataFrame
    index = ["1", "2", "3", "4", "5", "6", "7", "8"]
    columns = []
    frequency_matrix = np.zeros((len(index), len(top_melodies)))
    for i, melody in enumerate(top_melodies):
        for j, mode in enumerate(index):
            if mode in melody_frequencies[melody]:
                frequency_matrix[j, i] = melody_frequencies[melody][mode]
        # store counts
        columns.append(melody + "(" + str(int(np.sum(frequency_matrix[:, i]))) + ")")
        # normalize to sum up over modes to 1
        frequency_matrix[:, i] = frequency_matrix[:, i]/np.sum(frequency_matrix[:, i])


    df = DataFrame(frequency_matrix, index=index, columns=columns)

    return df