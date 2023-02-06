def wufpc_score(segmented_chants):
    """
    Weighted Unique Final Pitch Count Score

    Average number of melodies unique final pitches for single chant.
    Each chant is weighted by its number of segments.
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