

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