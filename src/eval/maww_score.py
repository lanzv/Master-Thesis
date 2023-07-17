from src.utils.loader import load_word_segmentations, load_phrase_segmentations

def maww_score(segmented_chants, chant_offset = 0):
    """
    Melody Aligned With Words score
    Frequency of words that end with end of any segment.
    maww_words = (1/word_segment_num)* Sum_word_segment_w_ends_at_same_index_as_any_segment(1)
    maww_segments = (1/segment_num)* Sum_word_segment_w_ends_at_same_index_as_any_segment(1)
    maww_average = (maww_words+maww_segments)/2

    Parameters
    ----------
    segmented_chants : list of lists of strings
        list of chants, each chant is represented as list of segments
    chant_offset : int
        offset of word segmented chants, to map the same chants of both, segmented_chants and word segmentations
    Returns
    -------
    maww_words : float
        maww score - how many words are aligned correctly with segments
    maww_segments : float
        maww score - how many segments are aligned correctly with words
    maww_average : float
        maww score - average of both maww_words and maww_segments
    """
    total_words = 0
    total_segments = 0
    correct_segments = 0
    words = load_word_segmentations()[chant_offset:]
    for chant_segments, word_segmentation in zip(segmented_chants, words):
        word_indices = set()
        i = 0
        for word in word_segmentation:
            i += len(word)
            word_indices.add(i)
        total_words += len(word_segmentation)
        i = 0
        for segment in chant_segments:
            i += len(segment)
            if i in word_indices:
                correct_segments += 1
        total_segments += len(chant_segments)

    # Compute all scores
    maww_words = float(correct_segments)/float(total_words)
    maww_segments = float(correct_segments)/float(total_segments)
    maww_average = (maww_words + maww_segments)/2.0

    return maww_words, maww_segments, maww_average


def mawp_score(model):
    """
    Melody Aligned With Phrase score
    Frequency of phrases that end with end of any segment.
    mjwp = (1/phrase_segment_num)* Sum_phrase_segment_p_ends_at_same_index_as_any_segment(1)

    Parameters
    ----------
    model : obj
        object that has "predict_segments" function that takes as an input list of strings (melodies ~ chants)
    Returns
    -------
    mjwp_score : float
        mjwp score
    """
    total_phrases = 0
    correct_segments = 0
    phrased_chants = load_phrase_segmentations()
    segmented_chants, perplexity = model.predict_segments([''.join(phrased_chant) for phrased_chant in phrased_chants])
    for chant_segments, phrase_segmentation in zip(segmented_chants, phrased_chants):
        phrase_indices = set()
        i = 0
        for phrase in phrase_segmentation:
            i += len(phrase)
            phrase_indices.add(i)
        i = 0
        for segment in chant_segments:
            i += len(segment)
            if i in phrase_indices:
                correct_segments += 1
        total_phrases += len(phrase_segmentation)

    return float(correct_segments)/float(total_phrases)