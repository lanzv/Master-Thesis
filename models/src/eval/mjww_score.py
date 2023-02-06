from src.utils.loader import load_word_segmentations

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
        total_segments += len(chant_segments)

        return correct_segments/total_segments