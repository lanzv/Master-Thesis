from src.utils.plotters import plot_mode_frequencies

def wtmf_score(segmented_chants, modes):
    """
    Weighted Top Mode Frequency score
    Average of top mode frequencies of all segments.
    wtmf = (1/all_segments_num) * Sum_s(occurences_of_segments_in_its_max_mode)

    Parameters
    ----------
    segmented_chants : list of lists of strings
        list of chants, each chant is represented as list of segments
    modes : list of chars
        list of modes of all chants
    Returns
    -------
    wtmf_score : float
        wtmf score
    """
    segment_frequencies = {}
    used_modes = set()
    # collect data
    for segments, mode in zip(segmented_chants, modes):
        for segment in segments:
            if not segment in segment_frequencies:
                segment_frequencies[segment] = {}
            if not mode in segment_frequencies[segment]:
                segment_frequencies[segment][mode] = 0
            segment_frequencies[segment][mode] += 1
            used_modes.add(mode)

    # compute score
    top_frequencies_sum = 0
    all_counts = 0
    for segment in segment_frequencies:
        # pick normalized top melody frequency
        count_sum = 0
        for mode in segment_frequencies[segment]:
            count_sum += segment_frequencies[segment][mode]
        max_mode = max(segment_frequencies[segment], key=segment_frequencies[segment].get)
        top_frequencies_sum += segment_frequencies[segment][max_mode]
        all_counts += count_sum


    return top_frequencies_sum/all_counts


def show_mode_segment_frequencies(segmented_chants, modes, mode_labels = ["1", "2", "3", "4", "5", "6", "7", "8"]):
    """
    Collect data about the segment occurences of each segment for each mode separatly.
    Then, it is ploted in charts compared with sum of all occurences.

    Parameters
    ----------
    segmented_chants : list of lists of strings
        list of segmented chants, each chant represented as list of string segments
    modes : list of strings
        list of chant modes
    mode_labels : list
        unique list of modes used in our dataset
    """
    print("----------------------------- Segments Occurences regarding the mode -----------------------------")
    frequencies_data = get_mode_segment_frequencies(segmented_chants, modes, mode_labels)
    plot_mode_frequencies(frequencies_data, mode_labels)
    print("--------------------------------------------------------------------------------------------------")


def get_mode_segment_frequencies(segmented_chants, modes, mode_labels = ["1", "2", "3", "4", "5", "6", "7", "8"]):
    """
    Collect data about the segment occurences of each segment for each mode separatly.
    Include also the sum of all occurences over all modes.
    The final segment list is orderd in the way of grouped segments by dominant modes and each group is sorted by
    all occurences over all modes. This is the list in the groups key of final dictionary frequencies_data.

    Parameters
    ----------
    segmented_chants : list of lists of strings
        list of segmented chants, each chant represented as list of string segments
    modes : list of strings
        list of chant modes
    mode_labels : list
        unique list of modes used in our dataset
    Returns
    -------
    frequencies_data : dict
        dictionary of 9 keys: "all", "1", "2", .. "8", each contains value of another dict with values
        "groups", "frequencies", where groups is a list of segments, frequencies is a list of frequencies over all given
        chants of specific segments at the same position in groups.
    """
    segment_frequencies = {}
    used_modes = set()
    # collect data
    for segments, mode in zip(segmented_chants, modes):
        for segment in segments:
            if not segment in segment_frequencies:
                segment_frequencies[segment] = {}
            if not mode in segment_frequencies[segment]:
                segment_frequencies[segment][mode] = 0
            segment_frequencies[segment][mode] += 1
            used_modes.add(mode)

    # set the segment labels order
    segment_labels = []
    grouped_mode_segments = {}
    for mode in mode_labels:
        grouped_mode_segments[mode] = []
    # group segments to its dominant modes
    for segment in segment_frequencies:
        max_mode = max(segment_frequencies[segment], key=segment_frequencies[segment].get)
        grouped_mode_segments[max_mode].append(segment)
    # sort groups
    for mode in mode_labels:
        sum_segment_frequencies = {}
        for segment in grouped_mode_segments[mode]:
            sum_segment_frequencies[segment] = 0
            for mode_label in mode_labels:
                if mode_label in segment_frequencies[segment]:
                    sum_segment_frequencies[segment] += segment_frequencies[segment][mode_label]
        segment_labels += sorted(grouped_mode_segments[mode],
                                 key=lambda x: sum_segment_frequencies[x],
                                 reverse=True)

    # Prepare data
    frequencies_data = {}
    frequencies = []
    for segment in segment_labels:
        frequency_sum = 0
        for mode in mode_labels:
            if mode in segment_frequencies[segment]:
                frequency_sum += segment_frequencies[segment][mode]
        frequencies.append(frequency_sum)
    frequencies_data["all"] = {
        "groups": segment_labels,
        "frequencies": frequencies
    }
    for mode in mode_labels:
        frequencies_mode = []
        for segment in segment_labels:
            if mode in segment_frequencies[segment]:
                frequencies_mode.append(segment_frequencies[segment][mode])
            else:
                frequencies_mode.append(0)
        frequencies_data[mode] = {
            "groups" : segment_labels,
            "frequencies": frequencies_mode
        }
    return frequencies_data