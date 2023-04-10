import numpy as np
from pandas import DataFrame
from src.utils.plotters import plot_mode_segment_statistics, plot_unique_segments_densities
from decimal import Decimal, ROUND_HALF_UP

def get_vocabulary_size(segmentation: list) -> int:
    """
    Function to get vocabulary of segmented chants and return the size of that vocabulary.

    Parameters
    ----------
    segmentation : list of list of strings
        list of train chants represented as list of string segments
        [["asda", "asdasd", "as", "ds"]]
    Returns
    -------
    vocabulary_size : int
        vocabulary size
    """
    vocabulary = set()
    for chant in segmentation:
        for segment in chant:
            vocabulary.add(segment)
    return len(vocabulary)

def get_average_segment_length(segmentation: list) -> float:
    """
    Function to get average segment length over all segmented chants.

    Parameters
    ----------
    segmentation : list of list of strings
        list of train chants represented as list of string segments
        [["asda", "asdasd", "as", "ds"]]
    Returns
    -------
    average_segment_length : float
        average segment length
    """
    all_segments = 0
    segment_length_sum = 0
    for chant in segmentation:
        for segment in chant:
            all_segments += 1
            segment_length_sum += len(segment)
    return float(segment_length_sum)/float(all_segments)

def show_mode_segment_statistics(segmentation, modes, mode_list = ["1", "2", "3", "4", "5", "6", "7", "8"]):
    """
    Plot statistics about segments in respect to single modes.
         - number of shared segments
         - number of distinct segments
         - number of vocabulary segments 
         - number of unqiue segments

    Parameters
    ----------
    segmentation : list of list of strings
        list of train chants represented as list of string segments
        [["asda", "asdasd", "as", "ds"]]
    modes : list of strings
        list of train modes
    mode_list : list of strings
        list of all unique modes we have in dataset
    """
    shared_dataframe, distinct_dataframe = get_2d_statistic_matrices(segmentation, modes, mode_list)
    unique_dataframe, vocabulary_sizes_dataframe = get_1d_statistic_matrices(segmentation, modes, mode_list)
    densities_dict = get_unique_segment_densities(segmentation, modes)
    print("------------- Modes Vocabulary Statistics -------------")
    plot_mode_segment_statistics(shared_dataframe, distinct_dataframe, vocabulary_sizes_dataframe, unique_dataframe)
    plot_unique_segments_densities(densities_dict, mode_list)
    print("-------------------------------------------------------")

def get_2d_statistic_matrices(segmentation: list, modes: list, mode_list = ["1", "2", "3", "4", "5", "6", "7", "8"]):
    """
    Get DataFrames of shared and distinct segments between each pair of mode's vocabularies.

    Parameters
    ----------
    segmentation : list of list of strings
        list of train chants represented as list of string segments
        [["asda", "asdasd", "as", "ds"]]
    modes : list of strings
        list of train modes
    mode_list : list of strings
        list of all unique modes we have in dataset
    Returns
    -------
    shared_df : DataFrame
        dataframe .. n x n table, n=len(mode_list), where each cell i,j is a number of shared segments between ith and jth modes
    distinct_df : DataFrame
        dataframe .. n x n table, n=len(mode_list), where each cell i,j is a number of distinct segments between ith and jth modes
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
    Get DataFrames of unique segments of each mode vocabulary (that is not in a vocabulary of any of other modes)
    and the other dataframe of vocabulary sizes of each mode.

    Parameters
    ----------
    segmentation : list of list of strings
        list of train chants represented as list of string segments
        [["asda", "asdasd", "as", "ds"]]
    modes : list of strings
        list of train modes
    mode_list : list of strings
        list of all unique modes we have in dataset
    Returns
    -------
    unique_df : DataFrame
        dataframe .. 1 x n table, n=len(mode_list), where each cell is a number of unique segments that are not in other mode's vocabularies
    vocab_df : DataFrame
        dataframe .. 1 x n table, n=len(mode_list), where each cell is a size of mode's vocabulary
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


def get_unique_segment_densities(segmentation: list, modes: list, mode_list = ["1", "2", "3", "4", "5", "6", "7", "8"]):
    """
    Compute densities of unique segments considering each position of modes (proportionally).
    The goal is to find out if there are more unique segments at beggining, or in the middle or at the end.

    Parameters
    ----------
    segmentation : list of list of strings
        list of train chants represented as list of string segments
        [["asda", "asdasd", "as", "ds"]]
    modes : list of strings
        list of train modes
    mode_list : list of strings
        list of all unique modes we have in dataset
    Returns
    -------
    densities : dict
        dict with keys of all modes, value is always a list of 400 elements, where 
        each has a percentage (%) of that in that position of chant were unique segments
        over all chants of that mode. Index 399 stands for 100%, 199 stands for 50%, etc..
    """
    # Preprocess data
    # Dictionary of all unique segments that occure in the specific mode
    mode_unique_segments = {}
    for mode in mode_list:
        mode_unique_segments[mode] = set()
    # Collect all unique segments of all modes
    for chant, mode in zip(segmentation, modes):
        for segment in chant:
            mode_unique_segments[mode].add(segment)
    # Create the final dataframe
    unique_values = {}
    for i in range(len(mode_list)):
        temp_vocab = mode_unique_segments[mode_list[i]].copy()
        for j in range(len(mode_list)):
            if not i == j:
                for segment in mode_unique_segments[mode_list[i]]:
                    if segment in mode_unique_segments[mode_list[j]] and segment in temp_vocab:
                        temp_vocab.remove(segment)
        unique_values[mode_list[i]] = temp_vocab

    # Get Density Data
    # Prepare Density
    densities = {}
    num_chants = {}
    densities_scale = 400
    for mode in mode_list:
        num_chants[mode] = 0
        densities[mode] = np.zeros((densities_scale)) # 100% in 400 cells -> 4 cells ~ 1%
    # Get Percentage distribution of unique segments
    for chant, mode in zip(segmentation, modes):
        chant_len = len(''.join(chant))
        actual_position = 0
        tone_size = float(densities_scale)/float(chant_len)
        segment_pointer = 0
        num_chants[mode] += 1
        for i in range(1, densities_scale+1):
            while i > Decimal((actual_position + len(chant[segment_pointer]))*tone_size).quantize(0, ROUND_HALF_UP):
                actual_position += len(chant[segment_pointer])
                segment_pointer += 1
            if chant[segment_pointer] in unique_values[mode]:
                densities[mode][i-1] += 1
    for mode in mode_list:
        densities[mode] /= num_chants[mode]
        densities[mode] *= 100
    return densities