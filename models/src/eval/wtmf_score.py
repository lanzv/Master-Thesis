import numpy as np



def tmf_score(segmented_chants, modes):
    """
    Weighted Top Mode Frequency score

    Average of top mode frequencies of all melodies.
    """
    melody_frequencies = {}
    used_modes = set()
    # collect data
    for segments, mode in zip(segmented_chants, modes):
        for segment in segments:
            if not segment in melody_frequencies:
                melody_frequencies[segment] = {}
            if not mode in melody_frequencies[segment]:
                melody_frequencies[segment][mode] = 0
            melody_frequencies[segment][mode] += 1
            used_modes.add(mode)

    # compute score
    top_frequencies_sum = 0
    all_counts = 0
    for melody in melody_frequencies:
        # pick normalized top melody frequency
        count_sum = 0
        for mode in melody_frequencies[melody]:
            count_sum += melody_frequencies[melody][mode]
        max_mode = max(melody_frequencies[melody], key=melody_frequencies[melody].get)
        top_frequencies_sum += melody_frequencies[melody][max_mode]
        all_counts += count_sum


    return top_frequencies_sum/all_counts