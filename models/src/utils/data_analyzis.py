import sys

def analyze_chants(chants: list, modes: list):
    """
    Analyze datasets and print basic statistics
        - number of chants
        - average+max+min segment lengths
        - mode distribution

    Parameters
    ----------
    chants : list of strings
        list of chants represented as string of notes
    modes : list of strings
        list of modes
    """
    print("-------------------------------- Dataset Analyzis --------------------------------")
    
    # Analyze chants
    print("\t Number of chants: {}".format(len(chants)))
    min_chant_length = sys.maxsize
    max_chant_length = 0
    chant_length_sum = 0
    for chant in chants:
        chant_length_sum += len(chant)
        if len(chant) < min_chant_length:
            min_chant_length = len(chant)
        if len(chant) > max_chant_length:
            max_chant_length = len(chant)
    print("\t Average chant length: {}".format(float(chant_length_sum)/float(len(chants))))
    print("\t Maximal chant length: {}".format(max_chant_length))
    print("\t Minimal chant length: {}".format(min_chant_length))

    # Analyze modes
    mode_statistics = {}
    for mode in modes:
        if not mode in mode_statistics:
            mode_statistics[mode] = 0
        mode_statistics[mode] += 1
    print("\t Mode distribution", mode_statistics)
    print("----------------------------------------------------------------------------------")



def analyze_phrases(chants: list):
    """
    Analyze gregobase dataset and print basic statistics
        - number of chants
        - average+minimal+maximal phrase length
        - number of phrases

    Parameters
    ----------
    chants : list of lists of strings
        phrase segmentation of each chant, e.g. [["asda", "ddd", "a", "aaa"], ["dddg", "khk"]]
    """
    print("-------------------------------- Dataset Analyzis --------------------------------")
    
    # Analyze chants
    print("\t Number of chants: {}".format(len(chants)))
    min_phrase_length = sys.maxsize
    max_phrase_length = 0
    phrase_length_sum = 0
    phrase_count = 0
    for chant in chants:
        for phrase in chant:
            phrase_count += 1
            phrase_length_sum += len(phrase)
            if len(phrase) < min_phrase_length:
                min_phrase_length = len(phrase)
            if len(phrase) > max_phrase_length:
                max_phrase_length = len(phrase)
    print("\t Number of phrases: {}".format(phrase_count))
    print("\t Average phrase length: {}".format(float(phrase_length_sum)/float(phrase_count)))
    print("\t Maximal phrase length: {}".format(max_phrase_length))
    print("\t Minimal phrase length: {}".format(min_phrase_length))
    
    
    print("----------------------------------------------------------------------------------")