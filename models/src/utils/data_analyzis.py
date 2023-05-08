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