import numpy as np
from pandas import DataFrame

def list2string(segmented_chants):
    """
    ["aaa", "bbb", "ccc"] -> ["aaa bbb ccc"]
    """
    string_segmentations = []
    for chant_segments in segmented_chants:
        string_segmentations.append(' '.join(chant_segments))
    return string_segmentations


def get_topmelodies_frequency(train_segmented_chants, train_modes,
                            test_segmented_chants, test_modes,
                            top_melodies: list, ignore_segments: bool = False):
    top_melodies = set(top_melodies)
    melody_frequencies = {}
    used_modes = set()
    # collect training data
    for segments, mode in zip(train_segmented_chants, train_modes):
        if ignore_segments:
            chant = ''.join(segments)
            for melody in top_melodies:
                if melody in chant:
                    if not melody in melody_frequencies:
                        melody_frequencies[melody] = {}
                    if not mode in melody_frequencies[melody]:
                        melody_frequencies[melody][mode] = 0
                    melody_frequencies[melody][mode] += 1
                    used_modes.add(mode)
        else:
            for segment in segments:
                if segment in top_melodies:
                    if not segment in melody_frequencies:
                        melody_frequencies[segment] = {}
                    if not mode in melody_frequencies[segment]:
                        melody_frequencies[segment][mode] = 0
                    melody_frequencies[segment][mode] += 1
                    used_modes.add(mode)

    # collect test data
    for segments, mode in zip(test_segmented_chants, test_modes):
        if ignore_segments:
            chant = ''.join(segments)
            for melody in top_melodies:
                if melody in chant:
                    if not melody in melody_frequencies:
                        melody_frequencies[melody] = {}
                    if not mode in melody_frequencies[melody]:
                        melody_frequencies[melody][mode] = 0
                    melody_frequencies[melody][mode] += 1
                    used_modes.add(mode)
        else:
            for segment in segments:
                if segment in top_melodies:
                    if not segment in melody_frequencies:
                        melody_frequencies[segment] = {}
                    if not mode in melody_frequencies[segment]:
                        melody_frequencies[segment][mode] = 0
                    melody_frequencies[segment][mode] += 1
                    used_modes.add(mode)



    # Create DataFrame
    index = list(used_modes)
    columns = []
    frequency_matrix = np.zeros((len(index), len(top_melodies)))
    for i, melody in enumerate(melody_frequencies):
        columns.append(melody)
        for j, mode in enumerate(index):
            if mode in melody_frequencies[melody]:
              frequency_matrix[j, i] = melody_frequencies[melody][mode]

    df = DataFrame(frequency_matrix, index=index, columns=columns)

    return df