from src.utils.loader import load_word_segmentations, load_phrase_segmentations

def mjww_score(segmented_chants):
    """
    Melody Justified With Words score

    Frequency of melodies that end with end of any word.
    """
    total_segments = 0
    correct_segments = 0
    words = load_word_segmentations()
    for chant_segments, word_segmentation in zip(segmented_chants, words):
        word_indices = set()
        i = 0
        for word in word_segmentation:
            i += len(word)
            word_indices.add(i)
        i = 0
        for segment in chant_segments:
            i += len(segment)
            if i in word_indices:
                correct_segments += 1
        total_segments += len(word_segmentation)

    return float(correct_segments)/float(total_segments)


def mjwp_score(model):
    """
    Melody Justified With Phrases score

    Frequency of melodies that end with end of any word.
    """
    total_segments = 0
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
        total_segments += len(phrase_segmentation)

    return float(correct_segments)/float(total_segments)