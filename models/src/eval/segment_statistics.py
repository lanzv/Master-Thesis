import numpy as np
from pandas import DataFrame
from src.utils.plotters import plot_mode_segment_statistics

def get_vocabulary_size(segmentation: list) -> int:
    """
    segmentation is a  list of lists of segments
    [["asda", "asdasd", "as", "ds"]]
    """
    vocabulary = set()
    for chant in segmentation:
        for segment in chant:
            vocabulary.add(segment)
    return len(vocabulary)

def get_average_segment_length(segmentation: list) -> float:
    """
    segmentation is a list of lists of segments
    [["asda", "asdasd", "as", "ds"]]
    """
    all_segments = 0
    segment_length_sum = 0
    for chant in segmentation:
        for segment in chant:
            all_segments += 1
            segment_length_sum += len(segment)
    return float(segment_length_sum)/float(all_segments)

def show_mode_segment_statistics(segmentation, modes, mode_list = ["1", "2", "3", "4", "5", "6", "7", "8"]):
    shared_dataframe, distinct_dataframe = get_2d_statistic_matrices(segmentation, modes, mode_list)
    unique_dataframe, vocabulary_sizes_dataframe = get_1d_statistic_matrices(segmentation, modes, mode_list)

    print("------------- Modes Vocabulary Statistics -------------")
    plot_mode_segment_statistics(shared_dataframe, distinct_dataframe, vocabulary_sizes_dataframe, unique_dataframe)
    print("-------------------------------------------------------")

def get_2d_statistic_matrices(segmentation: list, modes: list, mode_list = ["1", "2", "3", "4", "5", "6", "7", "8"]):
    """
    segmentation is a list of lists of segments
    [["asda", "asdasd", "as", "ds"]]
    """
    # Dictionary of all unique segments that occure in the specific mode
    mode_unique_segments = {}
    for mode in mode_list:
        mode_unique_segments[mode] = set()

    # Collect all unique segments of all modes
    for chant, mode in zip(segmentation, modes):
        for segment in chant:
            mode_unique_segments[mode].add(segment)

    # Create the final dataframe
    index = mode_list.copy()
    columns = mode_list.copy()
    shared_segments = np.zeros((len(index), len(columns)))
    distinct_segments = np.zeros((len(index), len(columns)))
    for i in range(len(mode_list)):
        for j in range(i+1, len(mode_list)):
            count = 0
            for segment in mode_unique_segments[mode_list[i]]:
                if segment in mode_unique_segments[mode_list[j]]:
                    count += 1
            shared_segments[j, i] = count
            shared_segments[i, j] = count
            distinct_segments[j, i] = len(mode_unique_segments[mode_list[i]]) + len(mode_unique_segments[mode_list[j]]) - 2*count
            distinct_segments[i, j] = len(mode_unique_segments[mode_list[i]]) + len(mode_unique_segments[mode_list[j]]) - 2*count
    shared_df = DataFrame(shared_segments, index=index, columns=columns)
    distinct_df = DataFrame(distinct_segments, index=index, columns=columns)
    return shared_df, distinct_df


def get_1d_statistic_matrices(segmentation: list, modes: list, mode_list = ["1", "2", "3", "4", "5", "6", "7", "8"]):
    """
    segmentation is a list of lists of segments
    [["asda", "asdasd", "as", "ds"]]
    """
    # Dictionary of all unique segments that occure in the specific mode
    mode_unique_segments = {}
    for mode in mode_list:
        mode_unique_segments[mode] = set()

    # Collect all unique segments of all modes
    for chant, mode in zip(segmentation, modes):
        for segment in chant:
            mode_unique_segments[mode].add(segment)
    

    # Create the final dataframe
    index = mode_list.copy()
    unique_segments = np.zeros((len(index)))
    vocab_segments = np.zeros((len(index)))
    for i in range(len(mode_list)):
        vocab_segments[i] = len(mode_unique_segments[mode_list[i]])
        temp_vocab = mode_unique_segments[mode_list[i]].copy()
        for j in range(len(mode_list)):
            if not i == j:
                for segment in mode_unique_segments[mode_list[i]]:
                    if segment in mode_unique_segments[mode_list[j]] and segment in temp_vocab:
                        temp_vocab.remove(segment)
        unique_segments[i] = len(temp_vocab)
    unique_df = DataFrame(unique_segments, index=index)
    vocab_df = DataFrame(vocab_segments, index=index)
    return unique_df, vocab_df