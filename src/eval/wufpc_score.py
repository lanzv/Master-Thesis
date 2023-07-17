def wufpc_score(segmented_chants):
    """
    Weighted Unique Final Pitch Count Score

    Average number of segment unique final pitches for single chant.
    Each chant is weighted by its number of segments.
    Final score is computed as
    wufpc = (1/(total_segment_num)) * Sum_c(unique_final_pitches_num_c*num_segments_c)

    Parameters
    ----------
    segmented_chants : list of lists of strings
        list of chants, each chant is represented as list of segments
    Returns
    -------
    wufpc_score : float
        wufpc score
    """
    total_pitches = 0 # weighted pitches of each chant summed together
    total_segments = 0 # weighted total chants
    for chant_segments in segmented_chants:
        final_pitches = set()
        for segment in chant_segments:
            final_pitches.add(segment[-1])
        total_segments += len(chant_segments)
        total_pitches += (len(chant_segments)*len(final_pitches))
    return total_pitches/total_segments